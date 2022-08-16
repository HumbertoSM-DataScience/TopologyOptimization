def random_d4_transform(x_batch, y_batch):
    '''Apply random transformation from D4 symmetry group
    # Arguments
        x_batch, y_batch: input tensors of size `(batch_size, height, width, any)`
    '''
    batch_size = x_batch.shape[0]
    
    # horizontal flip
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = x_batch[mask, :, ::-1]
    y_batch[mask] = y_batch[mask, :, ::-1]

    # vertical flip
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = x_batch[mask, ::-1]
    y_batch[mask] = y_batch[mask, ::-1]
    
    # 90* rotation
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = np.swapaxes(x_batch[mask], 1, 2)
    y_batch[mask] = np.swapaxes(y_batch[mask], 1, 2)

    return x_batch, y_batch