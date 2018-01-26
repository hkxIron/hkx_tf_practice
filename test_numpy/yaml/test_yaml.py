from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
class YParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)
if __name__ == "__main__":
    hparams = YParams('conf.yaml', 'large_hidden')
    print(hparams.num_hidden)  # print 1024
    print(hparams)  # print 1024
