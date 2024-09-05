<div align="center">
<h2 align="center">
   <b>Iterative Graph Alignment</b>
</h2>

<div>
  <a target="_blank" href="https://scholar.google.com.sg/citations?user=GqZfs_IAAAAJ&hl=en">Fangyuan&nbsp;Yu</a><sup>1</sup>,
  <a target="_blank">Hardeep&nbsp;Arora</a><sup>1</sup>,
  <a target="_blank">Matt&nbsp;Johnson</a><sup>1</sup>
</div>

<br />
<sup>1</sup>Temus&nbsp;&nbsp;&nbsp;
<br />
<div align="center">
    <a href="https://arxiv.org/abs/2408.16667" target="_blank">
</div>
</div>

![IGA](https://github.com/user-attachments/assets/e78350af-64b0-4bdb-90bc-c93b8eae96bb)
Iterative Graph Alignment (IGA) is an annotation-free alignment algorithm. A teacher model (VLM) iteratively generates logical
graphs and reference answers using Iterative Graph Prompting (IGP). A student model (LLM)
reviews its responses against these reference answers to identify hard cases where representation gaps
exist. The student then collaborates with helper models to explore diverse ways to respond to these
challenging queries by taking hints from the logical graphs and reference answers, before fine-tuning
on the collected insights and proceed to the next iteration.

## :new: Updates
- [08/2024] [arXiv Preprint] https://arxiv.org/abs/2408.16667 
- [08/2024] [Dataset Release] (https://github.com/fangyuan-ksgk/RuleEval) Rule-adherance ability evaluation suite.

<br />
<br />


Install Dependencies
```shell
bash set.sh
```

IGP prompting
```python
python -m script.reason
```

SAIL training 
```python
python -m script.iter
```

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@misc{yu2024iterativegraphalignment,
      title={Iterative Graph Alignment}, 
      author={Fangyuan Yu and Hardeep Singh Arora and Matt Johnson},
      year={2024},
      eprint={2408.16667},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2408.16667}, 
}
```
