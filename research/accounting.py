"""Record allocated CPU/GPU hours across successful, failed and cancelled jobs."""
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    ids=sorted(set(json.loads((ROOT/'research/jobs.json').read_text()).values()))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','login.hpc.virginia.edu',
        'sacct -n -j '+','.join(ids)+' --format=JobID,State,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode -P'],
        capture_output=True,text=True,check=True,timeout=45)
    rows=[]
    for line in r.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<6 or '.' in fields[0]:
            continue
        jid,state,elapsed,cpus,tres,code=fields[:6]
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        seconds=int(elapsed or 0)
        rows.append(dict(job=jid,state=state,seconds=seconds,cpus=int(cpus or 0),
                         gpus=int(match[1]) if match else 0,exit_code=code))
    report=dict(jobs=rows,cpu_core_hours=sum(x['seconds']*x['cpus'] for x in rows)/3600,
                gpu_hours=sum(x['seconds']*x['gpus'] for x in rows)/3600,
                caps=dict(cpu_core_hours=32,gpu_hours=24,storage_gib=20),
                note='Allocated wall time, including failed/cancelled work; excludes unstarted reservations. Service-unit charges differ.')
    dest=ROOT/'research/resource-accounting.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='jobs'},indent=2))

if __name__=='__main__':
    main()
