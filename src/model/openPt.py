import torch


def inspect_pt_file(file_path):
    try:
        # 加载 .pt 文件内容
        data = torch.load(file_path)

        # 打印文件内容的整体类型
        # print(f"questions: {data['questions']}\n")
        # print(f"contexts: {data['contexts'][33]}\n")
        print(f"retrieved_contexts: {data['retrieved_contexts']}\n")

        retrieved_contexts = data['retrieved_contexts']
        non_empty_count = sum(1 for context in retrieved_contexts if context != '')

        print(f"Number of non-empty retrieved contexts: {non_empty_count}")

    except Exception as e:
        print(f"Failed to load the .pt file. Error: {e}")


# 替换为你的 .pt 文件路径
# file_path = "../dataset/dataset/pubmed_qa/cache/pubmed_clear_encoded_graph.pt"
file_path = "../dataset/dataset/pubmed_qa/cache/3_labels/pubmed_bm25_encoded_graph.pt"
inspect_pt_file(file_path)
