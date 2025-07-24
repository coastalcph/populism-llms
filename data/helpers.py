from datasets import concatenate_datasets


def augment_dataset(dataset, upsample_ratio=5):
    pop_dataset = dataset.filter(lambda el:  el['pop_code'] != 0)
    pop_dataset = concatenate_datasets([pop_dataset for _ in range(upsample_ratio)])
    dataset = concatenate_datasets([dataset, pop_dataset])

    return dataset
