import yaml
from omegaconf import OmegaConf
from ibq_modules.ibqgan import load_vq_gan

yaml_path = './ibq_modules/ibq_config.yaml'

config = OmegaConf.load(yaml_path)

model = load_vq_gan(config)
print(model)


## A simple inference code is like:
#         image = np.array(image).astype(np.uint8)
#         image = (image/127.5 - 1.0).astype(np.float32)
#         images = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).to(device)
#         quant, qloss, (_, _, indices) = model.encode(images)
#         reconstructed_images = model.decode(quant)
