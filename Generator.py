import json
import tempfile

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from chatette.facade import Facade


class Generator:

    def __init__(self, template, output_file, augmentation_factor=5):
        assert augmentation_factor >= 2
        self.augmentation_factor = augmentation_factor

        self.base_file = template
        self.output_file = output_file

        self.dataset = {}
        self.intents = {}

        self.character_augmenter = nac.OcrAug()
        self.word_augmenter = naw.ContextualWordEmbsAug()
        # self.back_translator = naw.BackTranslationAug()

    def _createDatasetFromTemplate(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            generator = Facade(
                master_file_path=self.base_file,
                output_dir_path=tmpdirname,
                force_overwriting=True
            )
            generator.run()

            with open(tmpdirname + '/train/output.json') as dataset:
                dataset = json.load(dataset)
        self.dataset = dataset

    def _parseDataset(self):
        for example in self.dataset['rasa_nlu_data']['common_examples']:
            if example['intent'] not in self.intents:
                self.intents[example['intent']] = []
            self.intents[example['intent']].append(example['text'])

    def _augmentDataset(self):
        self.dataset = {}
        for key, intent in self.intents.items():
            self.dataset[key] = []
            for text in intent:
                self.dataset[key].append(text)
                self.dataset[key] += self._augmentText(text)

    def _augmentText(self, text):
        #text = self.back_translator.augment(text)
        text = self.word_augmenter.augment(text, self.augmentation_factor - 1)
        text = self.character_augmenter.augment(text)
        return text

    def _saveDataset(self):
        try:
            with open(self.output_file, 'w') as output_file:
                json.dump(self.dataset, output_file)
        except:
            print("Could not save dataset")

    def run(self):
        self._createDatasetFromTemplate()
        self._parseDataset()
        self._augmentDataset()
        self._saveDataset()
