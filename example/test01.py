from transformers import GPT2LMHeadModel, AutoTokenizer, TextGenerationPipeline

# from transformers import GPT2LMHeadModel, AutoTokenizer
# save_dir = "E:/github_rep/chinese-poems/model/gpt2-chinese-lyric/"
# model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-lyric")
# tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-lyric")
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)

# 中文歌词生成
model = GPT2LMHeadModel.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-lyric")
token = AutoTokenizer.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-lyric")

text_generator = TextGenerationPipeline(model, token)
out = text_generator("我走在买无糖可乐的路上，你在我前面", max_length= 100, do_sample = True)
print(out)