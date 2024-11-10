import os
import argparse
import logging
import autogluon as ag
from autogluon import ImageClassification as task

#import logging
#logger = spark._jvm.org.apache.log4j
#logging.getLogger("py4j").setLevel(logging.ERROR)



dataset = task.Dataset('../input/cassava-leaf-disease-classification/train_images', label_file='../input/cassava-leaf-disease-classification/train_ag.csv')

test_dataset = task.Dataset('../input/cassava-leaf-disease-classification/test_images', train=False, scale_ratio_choice=[0.7, 0.8, 0.875])



time_limits = 60 * 60 * 1 #60 
savedir = '../results/ag_model/'
classifier = task.fit(dataset, time_limits=time_limits, verbose=True, epochs=10, output_directory=savedir, problem_type='multiclass', eval_metric='accuracy',ngpus_per_trial=1,batch_size=8, input_size=512, tricks=dict({'last_gamma': False, 'use_pretrained': True, 'use_se': True, 'mixup': False, 'mixup_alpha': 0.2, 'mixup_off_epoch': 0, 'label_smoothing': True, 'no_wd': False, 'teacher_name': None, 'temperature': 20.0, 'hard_weight': 0.5, 'batch_norm': True, 'use_gn': False}), net='efficientnet_b1')

print('Top-1 val acc: %.3f' % classifier.results['best_reward'])

classifier.save(savedir+'exp2.ag')
classifier.load(savedir+'exp2.ag')

inds, probs, probs_all = classifier.predict(test_dataset)
print (inds, probs, probs_all)


ag.utils.generate_prob_csv(test_dataset, probs_all, custom='submission')



