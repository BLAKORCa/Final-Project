
class RnnConfig:
    def __init__(self, embed_in=500*500, embed_out=32, hidden_size=32, num_layers=1, batch_first=True):
        self.embed_in = embed_in
        self.embed_out = embed_out
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

class CnnConfig:
    def __init__(self, in_channels=1, out_channels=8, kernel_size=15,
                 stride=1, padding=2, pooling=5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling = pooling

class TrainConfig:
    def __init__(self, EPOCH, LR, loss_function, optimizer):
        self.EPOCH = EPOCH
        self.LR = LR
        self.loss_function = loss_function
        self.optimizer = optimizer

if __name__ == '__main__':
    cnn1_cg = CnnConfig(in_channels=1, out_channels=8, kernel_size=30, stride=5, padding=0)
    print(cnn1_cg.__dict__)