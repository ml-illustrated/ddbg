from .toy_cnn import ToyCNNModel
from .embed2 import Embed2Model, Embed2ModelEmbedOnly
from .resnet_preact import ResnetPreactModel, ResnetPreact50Model, ResnetPreact50ModelEmbedOnly

model_name__model_class = dict(
    toycnn = ToyCNNModel,
    embed2 = Embed2Model,
    embed2embed_only = Embed2ModelEmbedOnly,
    resnet = ResnetPreactModel,
    resnet50 = ResnetPreact50Model,
    resnet50embed_only = ResnetPreact50ModelEmbedOnly,
)
