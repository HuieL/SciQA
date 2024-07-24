# Processing logic

Top-20 papers in each leader board are selected as root papers. Root papers and their cited papers are searched on arxiv. All found papers are used to construct the graph:

```
graph = Data(x, edge_index, arxiv_id, content, abstract, title, desc='description of this graph')
```
