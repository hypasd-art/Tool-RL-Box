---
license: apache-2.0
task_categories:
- text-generation
language:
- en
- zh
tags:
- synthetic
- tools
size_categories:
- 10K<n<100K
---
# ToolACE

ToolACE is an automatic agentic pipeline designed to generate Accurate, Complex, and divErse tool-learning data. 
ToolACE leverages a novel self-evolution synthesis process to curate a comprehensive API pool of 26,507 diverse APIs. 
Dialogs are further generated through the interplay among multiple agents, guided by a formalized thinking process. 
To ensure data accuracy, we implement a dual-layer verification system combining rule-based and model-based checks. 
More details can be found in our paper on arxiv: [*ToolACE: Winning the Points of LLM Function Calling*](https://arxiv.org/abs/2409.00920).

The model finetuned with ToolACE is released in [hf](https://huggingface.co/Team-ACE/ToolACE-8B), winning promising scores on the functional calling leaderboard [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard).

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/646735a98334813a7ae29500/bpqk5cjaa8S_XkaMHHT0H.jpeg)

### Citation

If you think ToolACE is useful in your work, please cite our paper:
```
@misc{liu2024toolacewinningpointsllm,
      title={ToolACE: Winning the Points of LLM Function Calling}, 
      author={Weiwen Liu and Xu Huang and Xingshan Zeng and Xinlong Hao and Shuai Yu and Dexun Li and Shuai Wang and Weinan Gan and Zhengying Liu and Yuanqing Yu and Zezhong Wang and Yuxian Wang and Wu Ning and Yutai Hou and Bin Wang and Chuhan Wu and Xinzhi Wang and Yong Liu and Yasheng Wang and Duyu Tang and Dandan Tu and Lifeng Shang and Xin Jiang and Ruiming Tang and Defu Lian and Qun Liu and Enhong Chen},
      year={2024},
      eprint={2409.00920},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.00920}, 
}
```