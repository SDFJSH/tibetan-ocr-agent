"""Run on original AutoDL instance before pushing weights to GitHub."""
import argparse
from cloud_crnn import CachedCRNN

p = argparse.ArgumentParser()
p.add_argument('--weights', default='models/best.pt')
p.add_argument('--image', required=True)
a = p.parse_args()
print(CachedCRNN(a.weights).recognize(a.image))
