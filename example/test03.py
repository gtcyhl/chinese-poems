from transformers import GPT2LMHeadModel, AutoTokenizer, TextGenerationPipeline

# 中文对联生成
model = GPT2LMHeadModel.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-couplet")
token = AutoTokenizer.from_pretrained(r"E:\github_rep\chinese-poems\model\gpt2-chinese-couplet")

# 需要使用gpu，device=0
text_generator = TextGenerationPipeline(model, token)
out = text_generator("[CLS]十口心思，思乡思国思社稷 -", max_length= 28, do_sample = True)
print(out)