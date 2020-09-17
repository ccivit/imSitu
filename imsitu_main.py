from baseline_crf import baseline_crf
import torch
import torch.utils.data as data
from PIL import Image
import os
import yaml
import sys
import csv
import ntpath
from PIL import Image

def is_image(path):
    try:
        im = Image.open(path)
        return True
    except IOError:
        return False

class single_image(data.Dataset):
    # partially borrowed from ImageFolder dataset, but eliminating the assumption about labels
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.ext)

    def get_images(self, dir):
        image = [dir]
        return image

##    def get_images(self,dir):
##        images = []
##        for target in os.listdir(dir):
##            f = os.path.join(dir, target)
##            if os.path.isdir(f):
##                continue
##            if self.is_image_file(f):
##              images.append(target)
##        return images

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # list all images
        self.ext = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]
        self.images = self.get_images(root)

    def __getitem__(self, index):
        _id = os.path.join(self.root)
#        _id = os.path.join(self.root, self.images[index])
        img = Image.open(_id).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, torch.LongTensor([index])

    def __len__(self):
        return len(self.images)


def find_sentence_by_verb(sentence_file,verb):
    with open(sentence_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if verb == row[0]:
                return row[1]
        return 0


def generate_sentences(imsitu_outputs,config):
    sentences = {}
    for frame in imsitu_outputs:
        sentence = find_sentence_by_verb(config['sentences_file'],frame['verb']).split()
        for i,word in enumerate(sentence):
            if word_is_all_caps(word):
                try:
                    code = frame['frames'][0][word.lower()]
                    sentence[i] = replace_code_with_value(code,config['translations_file'])
                except:
                    sentence[i] = "/" + str(word.lower())+ "/"
                if sentence[i] == '' and i != 2:
                    sentence[i-1] = ''
        print(sentence)
        sentence = ' '.join(sentence)
        sentence = " ".join(sentence.split())
        sentences[sentence] = frame['score']

    print(sentences)
    return sentences


def word_is_all_caps(string):
    return string.upper() == string


def replace_code_with_value(code,translations_file):
    with open(translations_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if code == row[0]:
                value = row[1].split(',')[0]
                return value
        return code


def split_path(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    return folders


def predict_human_readable (dataset_loader, simple_dataset, output_path, model, top_k, encoder, config):
    model.eval()
    print("predicting...")
    mx = len(dataset_loader)
    for i, (input, index) in enumerate(dataset_loader):
        print ("{}/{} batches".format(i+1,mx))
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())#, volatile = True)
        (scores,predictions)  = model.forward_max(input_var)
        human = encoder.to_situation(predictions)
        (b,p,d) = predictions.size()
        for _b in range(0,b):
            items = []
            offset = _b *p
        for _p in range(0, p):
            items.append(human[offset + _p])
            items[-1]["score"] = scores.data[_b][_p].item()
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        # print(simple_dataset)
        print(items)
        print('Saving imSitu results to',output_path)
        sentences = generate_sentences(items,config)
        yaml.dump(sentences, open(output_path, "w"), default_flow_style=False)


def strip_dir_structure(image_dir_path,pipeline_path):
    image_split_path = split_path(image_dir_path)
    return os.path.join(pipeline_path,image_split_path[2], image_split_path[1], image_split_path[0])


def load_config_file(config_file):
    with open(config_file, 'r') as stream:
        config_general = yaml.safe_load(stream)
        config = config_general['imsitu']
    return config['weights'],\
           config['nb_results'],\
           config['cnn_type'],\
           config['encoder'],\
           config_general['general']['pipeline_path'],\
           config['results'],\
           config


if __name__ == "__main__":
    image_dir_path = sys.argv[1]
    config_file = sys.argv[2]
    do_overwrite = False

    weights_file, top_k, cnn_type, encoding_file, pipeline, results_file, config = load_config_file(config_file)
    total_imgs = len(os.listdir(image_dir_path))
    output_dir = strip_dir_structure(image_dir_path,pipeline)
    print(output_dir)
    batch_size = 64 # 64 is default
    encoder = torch.load(encoding_file)
    print("creating model...")
    model = baseline_crf(encoder, cnn_type=cnn_type)
    print("loading model weights...")
    model.load_state_dict(torch.load(weights_file))
    model.cuda()

    for i,img_filename in enumerate(os.listdir(image_dir_path)):
        img_file = os.path.join(image_dir_path, img_filename)
        if is_image(img_file):
            print(img_filename)
            print(os.listdir(image_dir_path))
            output_path = os.path.join(output_dir, img_filename.split('.')[0],results_file)

            print('File',i,'/',total_imgs)
            print('Output filename:',output_path)

            img = single_image(img_file, model.dev_preprocess())
            image_loader = torch.utils.data.DataLoader(img, batch_size=batch_size, shuffle=False,num_workers=3)
            if do_overwrite and os.path.isfile(output_path):
                continue
            else:
                predict_human_readable(image_loader, img, output_path, model, top_k,encoder,config)
