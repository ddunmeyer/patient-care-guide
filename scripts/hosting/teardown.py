"""Purpose: remove the Patient Care Guide demo server created by deploy.py.

Dry run by default. Add --apply to delete.

    python scripts/hosting/teardown.py
    python scripts/hosting/teardown.py --apply

Deletes the instance (and its disk, which holds the saved records), releases the
Elastic IP, removes the security group, the IAM role, the two SSM parameters and
the GitHub deploy key. It leaves the EC2 Scheduler alone (other tagged instances
may use it too).
"""
import argparse
import json
import subprocess
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

MANIFEST = Path(__file__).resolve().parents[2] / ".patientcare-hosting.json"
REPO = "ddunmeyer/patient-care-guide"


def quietly(action, *codes):
    """Run a delete and ignore 'already gone' errors, so teardown can be re-run safely."""
    try:
        action()
    except ClientError as err:
        if err.response["Error"]["Code"] not in codes:
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="delete the resources (default is a dry run)")
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} not found: nothing recorded to remove.")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"Will remove: instance {m.get('instance_id', '(none)')} and its disk, Elastic IP "
          f"{m.get('public_ip', '(none)')}, security group {m.get('security_group_id', '(none)')}, "
          f"IAM role {m['role']}, SSM parameters {', '.join(m['parameters'])}, deploy key {m['deploy_key_id']}.")
    if not args.apply:
        print("Dry run only. Nothing was deleted. Add --apply to remove.")
        return

    session = boto3.Session(region_name=m["region"])
    ec2, iam, ssm = session.client("ec2"), session.client("iam"), session.client("ssm")

    if m.get("instance_id"):
        quietly(lambda: ec2.terminate_instances(InstanceIds=[m["instance_id"]]), "InvalidInstanceID.NotFound")
        quietly(lambda: ec2.get_waiter("instance_terminated").wait(InstanceIds=[m["instance_id"]]),
                "InvalidInstanceID.NotFound")
    if m.get("allocation_id"):
        quietly(lambda: ec2.release_address(AllocationId=m["allocation_id"]), "InvalidAllocationID.NotFound")
    if m.get("security_group_id"):
        quietly(lambda: ec2.delete_security_group(GroupId=m["security_group_id"]), "InvalidGroup.NotFound")

    role = m["role"]
    quietly(lambda: iam.remove_role_from_instance_profile(InstanceProfileName=role, RoleName=role), "NoSuchEntity")
    quietly(lambda: iam.delete_instance_profile(InstanceProfileName=role), "NoSuchEntity")
    quietly(lambda: iam.delete_role_policy(RoleName=role, PolicyName="patientcare-app-access"), "NoSuchEntity")
    quietly(lambda: iam.detach_role_policy(RoleName=role, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"),
            "NoSuchEntity")
    quietly(lambda: iam.delete_role(RoleName=role), "NoSuchEntity")

    for name in m["parameters"]:
        quietly(lambda: ssm.delete_parameter(Name=name), "ParameterNotFound")
    subprocess.run(["gh", "repo", "deploy-key", "delete", m["deploy_key_id"], "--repo", REPO], check=False)

    MANIFEST.unlink()
    print("Removed. The EC2 Scheduler project is untouched.")


if __name__ == "__main__":
    main()
