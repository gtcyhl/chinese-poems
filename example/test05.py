from transformers import GPT2LMHeadModel, AutoTokenizer, TextGenerationPipeline

# 中文白话文生成
model = GPT2LMHeadModel.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-cluecorpussmall")
token = AutoTokenizer.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-cluecorpussmall")

# 需要使用gpu，device=0
text_generator = TextGenerationPipeline(model, token)
out = text_generator("白日依山尽", max_length= 50, do_sample = True)
print(out)
