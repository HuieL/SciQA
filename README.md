# Trian the CLEAR model
`python -m src.model.CLEAR`

# Load dataset

Now run following line then we will have a cache graph that contains encoded contexts and retrieved topk contexts.

`python -m src.dataset.pubmedqa --retriever bm25 --topk 10`
