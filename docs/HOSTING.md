# Hosting the Patient Care Guide on AWS

The care guide runs on one small EC2 instance with a fixed public address. An EC2 scheduler (EventBridge Scheduler and two Lambda functions) starts it each morning and stops it each evening, so it costs nothing while nobody is using it.

## What runs where

```
Client's phone or laptop
        |  http://<Elastic IP>      (port 80, plain HTTP)
        v
EC2 t3.small "PatientCare-Demo"     tagged AutoSchedule=true
  Streamlit care_app.py, run as a systemd service
  SQLite records + 24 h cache on the instance's encrypted disk
        |  IAM role PatientCareAppRole (reads its own two SSM parameters, nothing else)
        |-- SSM Parameter Store (encrypted): /patient-care-guide/env, /patient-care-guide/deploy-key
        |-- GitHub (private repo, read-only deploy key)
        '-- OpenAI, Langfuse, PubMed, ClinicalTrials.gov, openFDA, MedlinePlus, Tavily

EventBridge Scheduler --> StartEC2Instances (8:00 AM) / StopEC2Instances (6:00 PM), America/Chicago
        acts on every instance tagged AutoSchedule=true     <-- from the EC2 Scheduler project
```

Every time the service starts (each morning, or after a reboot) it fetches the keys from SSM, pulls the latest `main` from GitHub and installs any new packages. So to update the server: push to `main`, then restart the service or wait for the next morning.

## How secrets are handled

- **API keys.** The app calls OpenAI and Langfuse. Their keys are stored encrypted in SSM Parameter Store, and the server reads them with its IAM role. They are never in git or in the instance's startup script.
- **Private repo.** The server clones with a read-only GitHub deploy key, also kept in SSM.

## Deploy

Sign in with `aws login` and `gh auth login`, make sure `.env` holds your keys, install the AWS libraries once (`.venv\Scripts\pip install boto3 "botocore[crt]"`), then:

```bash
python scripts/hosting/deploy.py            # dry run: prints the plan, creates nothing
python scripts/hosting/deploy.py --apply    # creates the resources
```

The first boot installs packages and takes about five to ten minutes. The script prints the address and saves the resource IDs to `.patientcare-hosting.json` (not committed).

**Changed a key?** Re-run the `put_parameter` step by tearing down and redeploying, or update `/patient-care-guide/env` in the SSM console and restart the service.

## Remove

```bash
python scripts/hosting/teardown.py --apply
```

This deletes the instance and its disk (including any saved records), the Elastic IP, the security group, the IAM role, the two SSM parameters and the GitHub deploy key. It leaves the EC2 Scheduler alone.

## Things to know

- **It is public and has no sign-in.** Anyone with the address can use the app and see Patient History. Use made-up patients only; this setup is for demos, not real patient data.
- **Plain HTTP.** Answers travel unencrypted. Real use would need a domain name and HTTPS.
- **It is off outside 8:00 AM to 6:00 PM Central.** To keep it on for an evening demo, set the instance's `DoNotStop` tag to `true`, and set it back afterwards.
- **Shell access** is through Session Manager (EC2 console → Connect → Session Manager). There is no SSH. Logs: `sudo journalctl -u patientcare -f`.
- **Restart the app:** `sudo systemctl restart patientcare` (this also pulls the latest code).
- **Cost.** About $0.50 a day for the instance while it runs, about $0.12 a day for the Elastic IP, a little for the 20 GB disk, plus OpenAI usage.
- **The scheduler acts on every tagged instance.** Any other instance tagged `AutoSchedule=true` in the account starts and stops on the same timetable.
