from transformers import GPT2LMHeadModel, AutoTokenizer, TextGenerationPipeline

# 中文古诗生成
model = GPT2LMHeadModel.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-couplet")
token = AutoTokenizer.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-couplet")

# 需要使用gpu，device=0
text_generator = TextGenerationPipeline(model, token)
out = text_generator("白日依山尽,[CLS]", max_length= 50, do_sample = True)
print(out)
