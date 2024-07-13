from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--test_epoch', type=str, default='latest', help='how many epochs to run')
        self.isTrain = False
        return parser
