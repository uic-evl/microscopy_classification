import os

def main():
    outputs = ['conv5_block3_out', 'conv5_block2_out', 'conv5_block1_out']
    #            'conv4_block6_out', 'conv4_block5_out', 'conv4_block4_out', 'conv4_block3_out', 'conv4_block2_out', 'conv4_block1_out',
    #            'conv3_block4_out', 'conv3_block3_out', 'conv3_block2_out', 'conv3_block1_out',
    #            'conv2_block2_out', 'conv2_block1_out']

    tuning = [ 'conv5_block3_out', 'conv5_block2_out', 'conv5_block1_out', 
               'conv4_block6_out', 'conv4_block5_out', 'conv4_block4_out', 'conv4_block3_out', 'conv4_block2_out', 'conv4_block1_out']
               #'conv3_block4_out', 'conv3_block3_out', 'conv3_block2_out', 'conv3_block1_out',
               #'conv2_block2_out', 'conv2_block1_out']


    model_name = 'resnet50'
    run_name = 'batch'
    batch_size = 32
    lr = 1e-5
    patience = 40
    dataset = 'ds1_clef16_microscopy_20'
    num_classes = 4
    optimizer = 'adam'
    epochs = 40
    
    for output in outputs:
        tun_idx = tuning.index(output)
        run_tuning = tuning[tun_idx:]
        run_tuning.insert(0, 'all')

        for ft in run_tuning:
            if ft == 'all':
                ft_name = 'all'
            else:
                ft_name = 'c%sb%s' % (ft[4], ft[11])

            name1 = '%s_c%sb%s_%s_20' % (run_name, output[4], output[11], ft_name)

            cmd = 'python train.py -epochs %d -model_name %s -run_name %s -batch_size %d -lr %f ' \
                  '-patience %d -dataset %s -num_classes %d -optimizer %s -resnet_output %s -fine_tune %s' \
                  % (epochs, model_name, name1, batch_size, lr, patience, dataset, num_classes, optimizer, output, ft)
            print(cmd)
            os.system(cmd)
            print('---------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    main()