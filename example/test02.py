from transformers import GPT2LMHeadModel, AutoTokenizer, TextGenerationPipeline

# 中文文言文生成
model = GPT2LMHeadModel.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-ancient")
token = AutoTokenizer.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-ancient")

# 需要使用gpu，device=0
text_generator = TextGenerationPipeline(model, token)
out = text_generator("蜀道之难", max_length= 100, do_sample = True)
print(out)