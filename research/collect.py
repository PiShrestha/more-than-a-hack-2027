"""Read cluster state and back up private artifacts locally; does not submit jobs."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'research/artifacts'
HOST = 'login.hpc.virginia.edu'
REMOTE_CODE = 'Documents/github/possession-research-20261116'
REMOTE_WORK = '/scratch/jpj8rf/possession-research-20261116'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ids = ','.join(json.loads((ROOT/'research/jobs.json').read_text()).values())
    command = ('sacct -j '+ids+' --format=JobID,JobName,State,ElapsedRaw,AllocCPUS,MaxRSS,AllocTRES,ExitCode -P; '
               'squeue -u "$USER" -o "%.20i %.22j %.8T %.10M %R"')
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,command],
                       capture_output=True,text=True,timeout=45,check=True)
    (DEST/'cluster-status.txt').write_text(r.stdout)
    print(r.stdout)
    for remote,local in [(REMOTE_CODE+'/logs/','logs/'),
                         (REMOTE_WORK+'/results/','results/'),
                         (REMOTE_WORK+'/data/','data/')]:
        p = DEST/local
        p.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','--partial','--exclude=*.part',
                        HOST+':'+remote,str(p)+'/'],check=True,timeout=1800)
    print('Private data and experiment backups:',DEST)

if __name__=='__main__':
    main()
