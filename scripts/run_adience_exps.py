from egs.adience import Config
from run import run_with_config

for method in [
    'UNIORD',
    'Liu',
    'SORD',
    'BeckhamBinomial',
    'DLDL'
]:
    Config.method = method
    run_with_config(Config)
