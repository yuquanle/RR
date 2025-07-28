from torch.utils.data import Dataset
import json

class CaseData(Dataset):
    def __init__(self, mode='train', train_file=None, valid_file=None, test_file=None):
        assert mode in ['train', 'valid', 'test'], f"mode should be set to the one of ['train', 'valid', 'test']"
        self.mode = mode
        self.dataset = []
        if mode == 'train':
            self.dataset = self._load_data(train_file)
            print(f'Number of training dataset: {len(self.dataset)}')
        elif mode == 'valid':
            self.dataset = self._load_data(valid_file)
            print(f'Number of validation dataset: {len(self.dataset)}')
        else:
            self.dataset = self._load_data(test_file)
            print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        features_content = self.dataset[idx]['fact'][:1536]
        if self.mode in ['train', 'valid']:
            labels_accu = self.dataset[idx]['accu']
            labels_law = self.dataset[idx]['law']
            labels_term = self.dataset[idx]['term']
            return features_content, labels_accu, labels_law, labels_term
        else:
            return features_content

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        features_content, labels_accu, labels_law, labels_term = zip(*batch)
        return features_content, labels_accu, labels_law, labels_term
    
    def _load_data(self, file_name):
        dataset = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                json_dict = json.loads(line)
                dataset.append(json_dict)
        return dataset

if __name__ == '__main__':
    train_file = './datasets/cail_small/process_data_train.json'
    valid_file = './datasets/cail_small/process_data_valid.json'
    test_file = './datasets/cail_small/process_data_test.json'
    training_data = CaseData(mode='train', train_file=train_file)
    print(len(training_data))
