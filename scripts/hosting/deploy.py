"""Purpose: host the Patient Care Guide on one EC2 instance that the EC2 Scheduler project starts and stops.

Dry run by default: prints the plan
and creates nothing. Add --apply to build.

    python scripts/hosting/deploy.py            # show the plan
    python scripts/hosting/deploy.py --apply    # create the resources

What --apply creates in the signed-in AWS account (us-east-1 unless AWS_REGION says otherwise):
  - A read-only GitHub deploy key on ddunmeyer/patient-care-guide (the repo is
    private, so the server needs it to clone and pull)
  - Two encrypted SSM parameters: /patient-care-guide/env (your local .env:
    OpenAI, Langfuse, Tavily keys) and /patient-care-guide/deploy-key
  - IAM role + instance profile PatientCareAppRole (read those two parameters,
    Session Manager access; nothing else)
  - Security group PatientCare-Demo-Web: port 80 open to the internet, no SSH
  - One t3.small instance tagged AutoSchedule=true, running the app as a service
  - One Elastic IP, so the address survives the nightly stop and morning start
and it makes sure the existing EC2-Lab start and stop schedules are ENABLED.

Needs: `aws login`, `gh auth login`, `pip install boto3 "botocore[crt]"`, and a filled-in .env.
Resource IDs are saved to .patientcare-hosting.json (not committed) for teardown.py.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[2]
REGION = os.getenv("AWS_REGION") or "us-east-1"
REPO = "ddunmeyer/patient-care-guide"
REPO_SSH = f"git@github.com:{REPO}.git"
PREFIX = "/patient-care-guide"            # SSM parameter names start with this

NAME = "PatientCare-Demo"
ROLE = "PatientCareAppRole"
SECURITY_GROUP = "PatientCare-Demo-Web"
INSTANCE_TYPE = "t3.small"
DISK_GB = 20                              # the app's packages (chromadb, onnxruntime...) need room
AMI_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
SCHEDULES = ["EC2-Lab-Start-Schedule", "EC2-Lab-Stop-Schedule"]
MANIFEST = ROOT / ".patientcare-hosting.json"
TAGS = [{"Key": "project", "Value": "patient-care-guide"}]


def app_policy(account: str) -> dict:
    """Everything the running server is allowed to do: read its own two parameters."""
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "ReadAppSettings", "Effect": "Allow", "Action": "ssm:GetParameter",
         "Resource": f"arn:aws:ssm:{REGION}:{account}:parameter{PREFIX}/*"},
    ]}


def user_data() -> str:
    # A Windows checkout may give the script CRLF line endings, which would break it on Linux.
    script = (Path(__file__).parent / "server_setup.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    for key, value in {"__REGION__": REGION, "__PREFIX__": PREFIX, "__REPO_SSH__": REPO_SSH}.items():
        script = script.replace(key, value)
    return script


def read_env() -> str:
    """The local .env, minus blank lines. It goes to SSM encrypted, never into git or user data."""
    env = ROOT / ".env"
    if not env.exists():
        raise SystemExit(".env not found. Copy .env.example to .env and fill in the keys first.")
    lines = [l.strip() for l in env.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if not any(l.startswith("OPENAI_API_KEY=") and len(l) > len("OPENAI_API_KEY=") for l in lines):
        raise SystemExit("OPENAI_API_KEY is empty in .env.")
    return "\n".join(lines) + "\n"


def add_deploy_key() -> tuple[str, str]:
    """Make a fresh SSH key pair and register the public half as a read-only deploy key.
    Returns (private key text, deploy key id)."""
    with tempfile.TemporaryDirectory() as tmp:
        key = Path(tmp) / "id_ed25519"
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", NAME, "-f", str(key), "-q"], check=True)
        title = f"{NAME} server (read-only)"
        subprocess.run(["gh", "repo", "deploy-key", "add", str(key) + ".pub", "--repo", REPO,
                        "--title", title], check=True, capture_output=True)
        private = key.read_text(encoding="utf-8")
    keys = json.loads(subprocess.run(["gh", "api", f"repos/{REPO}/keys"], check=True,
                                     capture_output=True, text=True).stdout)
    key_id = next(str(k["id"]) for k in keys if k["title"] == title)
    return private, key_id


def put_parameters(ssm, env_text: str, private_key: str) -> None:
    for name, value, about in [(f"{PREFIX}/env", env_text, "Patient Care Guide .env (API keys)"),
                               (f"{PREFIX}/deploy-key", private_key, "Read-only GitHub deploy key")]:
        ssm.put_parameter(Name=name, Value=value, Type="SecureString", Overwrite=True, Description=about)
        ssm.add_tags_to_resource(ResourceType="Parameter", ResourceId=name, Tags=TAGS)


def create_role(iam, account: str) -> None:
    trust = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Tags=TAGS,
                        Description="Patient Care Guide demo server: read its own SSM settings")
    except ClientError as err:
        if err.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
    iam.put_role_policy(RoleName=ROLE, PolicyName="patientcare-app-access",
                        PolicyDocument=json.dumps(app_policy(account)))
    # Session Manager gives shell access for debugging without opening SSH or storing a key.
    iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    try:
        iam.create_instance_profile(InstanceProfileName=ROLE)
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE, RoleName=ROLE)
    except ClientError as err:
        if err.response["Error"]["Code"] != "EntityAlreadyExists":
            raise


def create_security_group(ec2) -> str:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    found = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SECURITY_GROUP]},
                                                  {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if found:
        return found[0]["GroupId"]
    group = ec2.create_security_group(
        GroupName=SECURITY_GROUP, VpcId=vpc, Description="Patient Care Guide demo: HTTP from anywhere, no SSH",
        TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS + [{"Key": "Name", "Value": SECURITY_GROUP}]}])
    ec2.authorize_security_group_ingress(GroupId=group["GroupId"], IpPermissions=[{
        "IpProtocol": "tcp", "FromPort": 80, "ToPort": 80,
        "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Public HTTP for the client demo"}]}])
    return group["GroupId"]


def launch(ec2, ami: str, group_id: str) -> str:
    instance_tags = TAGS + [{"Key": "Name", "Value": NAME}, {"Key": "AutoSchedule", "Value": "true"},
                            {"Key": "ManagedBy", "Value": "EC2Scheduler"}, {"Key": "Environment", "Value": "Demo"},
                            {"Key": "DoNotStop", "Value": "false"}]
    for attempt in range(8):  # a brand-new instance profile takes a few seconds to become usable
        try:
            result = ec2.run_instances(
                ImageId=ami, InstanceType=INSTANCE_TYPE, MinCount=1, MaxCount=1,
                IamInstanceProfile={"Name": ROLE}, SecurityGroupIds=[group_id], UserData=user_data(),
                MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                                      "Ebs": {"VolumeSize": DISK_GB, "VolumeType": "gp3",
                                              "Encrypted": True, "DeleteOnTermination": True}}],
                TagSpecifications=[{"ResourceType": "instance", "Tags": instance_tags},
                                   {"ResourceType": "volume", "Tags": TAGS}])
            return result["Instances"][0]["InstanceId"]
        except ClientError as err:
            if "Invalid IAM Instance Profile" not in str(err) or attempt == 7:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def enable_schedules(scheduler) -> None:
    """Make sure the EC2 Scheduler project's start and stop schedules are on, keeping everything else as is."""
    keep = ("ScheduleExpression", "ScheduleExpressionTimezone", "FlexibleTimeWindow", "Target", "Description",
            "GroupName", "StartDate", "EndDate", "KmsKeyArn", "ActionAfterCompletion")
    for name in SCHEDULES:
        current = scheduler.get_schedule(Name=name)
        if current["State"] != "ENABLED":
            scheduler.update_schedule(Name=name, State="ENABLED", **{k: current[k] for k in keep if k in current})
        print(f"  {name}: {current['ScheduleExpression']} {current.get('ScheduleExpressionTimezone', '')} -> ENABLED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="create the resources (default is a dry run)")
    parser.add_argument("--no-schedules", action="store_true", help="leave the start/stop schedules as they are")
    args = parser.parse_args()

    env_text = read_env()
    session = boto3.Session(region_name=REGION)
    identity = session.client("sts").get_caller_identity()
    account = identity["Account"]
    ec2, iam, ssm = session.client("ec2"), session.client("iam"), session.client("ssm")

    print(f"Signed in as {identity['Arn'].split('/')[-1]} in {REGION}.")
    print(f"Plan: read-only deploy key on {REPO}; SSM parameters {PREFIX}/env and {PREFIX}/deploy-key;")
    print(f"      IAM role {ROLE}; security group {SECURITY_GROUP} (port 80 open to the internet);")
    print(f"      one {INSTANCE_TYPE} named {NAME} tagged AutoSchedule=true; one Elastic IP;")
    print(f"      .env keys sent (values hidden): {', '.join(l.split('=')[0] for l in env_text.splitlines())};")
    print(f"      enable schedules: {'no' if args.no_schedules else ', '.join(SCHEDULES)}.")
    if not args.apply:
        print("Dry run only. Nothing was created. Add --apply to build.")
        return
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} already exists: a server is already deployed. Run teardown.py first.")

    private_key, key_id = add_deploy_key()
    manifest = {"region": REGION, "deploy_key_id": key_id, "parameters": [f"{PREFIX}/env", f"{PREFIX}/deploy-key"],
                "role": ROLE}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    put_parameters(ssm, env_text, private_key)
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    create_role(iam, account)
    group_id = create_security_group(ec2)
    manifest["security_group_id"] = group_id
    instance_id = launch(ec2, ami, group_id)
    manifest["instance_id"] = instance_id
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Launched {instance_id}. Waiting for it to run...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])

    address = ec2.allocate_address(Domain="vpc", TagSpecifications=[
        {"ResourceType": "elastic-ip", "Tags": TAGS + [{"Key": "Name", "Value": NAME}]}])
    ec2.associate_address(InstanceId=instance_id, AllocationId=address["AllocationId"])
    manifest.update(allocation_id=address["AllocationId"], public_ip=address["PublicIp"],
                    url=f"http://{address['PublicIp']}")
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.no_schedules:
        print("EC2 Scheduler schedules:")
        enable_schedules(session.client("scheduler"))

    print(f"\nServer address: {manifest['url']}")
    print("First boot installs Python packages and takes about five to ten minutes before the page answers.")
    print("The scheduler stops it at 6:00 PM and starts it at 8:00 AM, America/Chicago.")


if __name__ == "__main__":
    main()
