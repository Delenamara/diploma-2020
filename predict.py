import matplotlib.pyplot as plt

from model import TPN
from data_loader import *
from util import *

class PalletePredictor(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def prepare_dict(self):
        input_dict = Dictionary()
        src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()

        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])
        return input_dict

    def build_model(self):
        # Data loader.
        self.input_dict = self.prepare_dict()

        # Load pre-trained GloVe embeddings.
        emb_file = os.path.join('./data', 'Color-Hex-vf.pth')
        if os.path.isfile(emb_file):
            W_emb = torch.load(emb_file)
        else:
            W_emb = load_pretrained_embedding(self.input_dict.word2index,
                                              '../data/glove.840B.300d.txt',
                                              300)
            W_emb = torch.from_numpy(W_emb)
            torch.save(W_emb, emb_file)
        W_emb = W_emb.to(self.device)

        self.encoder = TPN.EncoderRNN(self.input_dict.n_words, 150, 1, 0.2, W_emb).to(self.device)
        self.G_TPN = TPN.AttnDecoderRNN(self.input_dict, 150, 1, 0.2).to(self.device)


        self.encoder.eval()
        self.G_TPN.eval()

    def load_model(self):
        encoder_path = os.path.join('./models/TPN', '{}_G_encoder.ckpt'.format(500))
        G_TPN_path = os.path.join('./models/TPN', '{}_G_decoder.ckpt'.format(500))

        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.G_TPN.load_state_dict(torch.load(G_TPN_path, map_location=lambda storage, loc: storage))

    def get_pallete(self, palette_name):
        self.load_model()

        palette_name = palette_name.split(' ')
        src_seq =[0] * self.input_dict.max_len
        for i, word in enumerate(palette_name):
            src_seq[i] = self.input_dict.word2index[word]
        txt_embeddings = torch.LongTensor([src_seq, src_seq])

        # Compute text input size (without zero padding).
        batch_size = 2
        nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
        each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

        # Prepare test data.
        txt_embeddings = txt_embeddings.to(self.device)
        fake_rgb_pallete = []

        for j in range(5):

            palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
            fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

            # ============================== Text-to-Palette ==============================#
            # Condition for the generator.
            encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
            encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

            # Generate color palette.
            for i in range(5):
                palette, decoder_context, decoder_hidden = self.G_TPN(palette,
                                                                         decoder_hidden.squeeze(0),
                                                                         encoder_outputs,
                                                                         each_input_size,
                                                                         i)
                fake_palettes[:, 3 * i:3 * (i + 1)] = palette
            # ================================ Save Results ================================#
            # Input text.
            input_text = ''
            for idx in txt_embeddings[0]:
                if idx.item() == 0: break
                input_text += self.input_dict.index2word[idx.item()] + ' '

            # Save palette generation results.
            fig1, axs1 = plt.subplots(nrows=1, ncols=5)
            axs1[0].set_title(input_text + '{}'.format(j+1))
            fake_lab_pallete = []
            for k in range(5):
                lab_fake = np.array([fake_palettes.data[0][3 * k],
                                fake_palettes.data[0][3 * k + 1],
                                fake_palettes.data[0][3 * k + 2]], dtype='float64')
                fake_lab_pallete.append(lab_fake)
                rgb = lab2rgb_1d(lab_fake)
                fake_rgb_pallete.append(rgb)
                print(fake_lab_pallete)
                axs1[k].imshow([[rgb]])
                axs1[k].axis('off')
            fig1.savefig('{}.jpg'.format(input_text.strip()))
        # TODO: add conver to hex here
        return fake_rgb_pallete