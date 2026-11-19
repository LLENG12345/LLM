"""Register this checkout's src in the active project venv without downloads."""
from pathlib import Path
import sys
import sysconfig

root = Path(__file__).resolve().parents[1]
environment = Path(sys.prefix).resolve()
site_packages = Path(sysconfig.get_path('purelib')).resolve()
if sys.prefix == sys.base_prefix or not site_packages.is_relative_to(environment):
    raise SystemExit('Use the project virtual environment interpreter; system Python is not modified')
source = root / 'src'
if not (source / 'ntl/__init__.py').is_file():
    raise SystemExit('Missing src/ntl package')
target = site_packages / 'name_that_llm_local.pth'
payload = str(source.resolve()) + '\n'
if target.exists() and target.read_text(encoding='utf-8') != payload:
    raise SystemExit(f'{target} already points at another checkout; inspect it before changing it')
target.write_text(payload, encoding='utf-8', newline='\n')
print('LOCAL_SOURCE_REGISTERED', target)
print('Restart existing Python/Jupyter processes before importing ntl')
