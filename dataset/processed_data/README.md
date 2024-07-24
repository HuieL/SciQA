# Processing logic

Top-20 papers in each leader board are selected as root papers. Root papers and their cited papers are searched on arxiv. All found papers are used to construct the graph:

```
graph = Data(x, edge_index, arxiv_id, content, abstract, title, desc='description of this graph')
```

You can find graphs here: https://drive.google.com/drive/folders/13hU7NQY_7TYmuk09UJad0CUbTxcbjlYy?usp=drive_link
You should use file of questions here.
