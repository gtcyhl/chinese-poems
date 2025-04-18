from torch.utils.data import Dataset


class ChineseDataset(Dataset):
    def __init__(self):
        with open("data/chinese_poems.txt", encoding="utf-8") as f:
            lines = f.readlines()
        # 去除每数据的前后空格
        lines = [i.strip() for i in lines]
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        return self.lines[item]


if __name__ == "__main__":
    dataset = ChineseDataset()
    print(len(dataset), dataset[-1])
