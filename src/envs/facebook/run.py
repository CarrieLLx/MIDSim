"""Launch facebook using the shared project runtime."""
from pathlib import Path
import os,sys,runpy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'src'))
args=sys.argv[1:]
defaults={'--config':str(ROOT/'config/config_facebook.json'),'--model_config':str(ROOT/'config/model_config.json'),'--env':'facebook','--mode':'single'}
for key,value in defaults.items():
 if not any(a==key or a.startswith(key+'=') for a in args):args.extend([key,value])
sys.argv=[str(ROOT/'src/main.py'),*args]
runpy.run_path(sys.argv[0],run_name='__main__')
