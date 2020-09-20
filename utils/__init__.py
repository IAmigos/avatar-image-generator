import json


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file


def init_logger(log_file=None, log_dir=None):

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'

    if log_dir is None:
        log_dir = '~/temp/log/'

    if not os.path.exists(log_dir):
        print("Creating dir")
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, log_file)

    print('log file path:' + log_file)

    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        format=fmt)

    return logging


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print('Applied to:: ', m.__class__.__name__)
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
