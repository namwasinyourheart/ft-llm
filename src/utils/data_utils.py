def show_sample(dataset,split, idx):
  for key in dataset[split][idx].keys():
    print(f"{key}:")
    print(dataset[split][idx][key])
    print()