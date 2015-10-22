#!/bin/bash

####################
# Modality-Biased Log-Bilinear Model 
####################
lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig

./train_mplm.py --lang_list en --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en.log

./best_system.py ../work/mlbl_b_orig/en.log
### Epoch 77 Perplexity 5.49438582118
### Test on English
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/en/77 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.45796 perplexity: 5.49438582118
# Test cost mean: 2.45856 perplexity: 5.49667172657

### Test on French
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/en/77 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 5.78292 perplexity: 55.0596380766
# Test cost mean: 5.77962 perplexity: 54.9337050481
####################################################################################################

./train_mplm.py --lang_list fr --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/fr.log
./best_system.py --log_file ../work/mlbl_b_orig/fr.log
### Epoch 61 Perplexity 6.01621639069
### Test on English
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/fr/61 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/p
ron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 5.47481 perplexity: 44.471573741
# Test cost mean: 5.46099 perplexity: 44.0476398239

### Test on French
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/fr/61 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.58886 perplexity: 6.01621639069
# Test cost mean: 2.59189 perplexity: 6.02888831616
####################################################################################################


./train_mplm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr.log
./best_system.py --log_file ../work/mlbl_b_orig/en_fr.log
### Epoch 96 Perplexity 5.97537230968

### Test on English
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/en_fr/96 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.5214 perplexity: 5.74137733422
# Test cost mean: 2.52476 perplexity: 5.75478929738

### Test on French
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/en_fr/96 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.60562 perplexity: 6.0865317659
# Test cost mean: 2.60788 perplexity: 6.09606970902

####################################################################################################


lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/no_lang_vector

./train_mplm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_no_lang_vector.log

./best_system.py --log_file ../work/mlbl_b_orig/no_lang_vector/en_fr_no_lang_vector.log 
### Epoch 96 Perplexity 6.4283478754

### Test on English
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/no_lang_vector/en_fr/96 --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat. --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.72841 perplexity: 6.62723092528
# Test cost mean: 2.73092 perplexity: 6.63879876743

### Test on French
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/no_lang_vector/en_fr/96 --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat. --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.66413 perplexity: 6.33847149013
# Test cost mean: 2.66621 perplexity: 6.34761428996

####################################################################################################

lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/lang_id

#tmux1
./train_mplm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_lang_id.log

./best_system.py --log_file ../work/mlbl_b_orig/no_lang_vector/en_fr_lang_id.log
### 

### Test on English
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/no_lang_vector/en_fr/96 --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat. --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# 

### Test on French
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/no_lang_vector/en_fr/96 --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat. --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# 

####################################################################################################

####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
####################

lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid


./train_mnlm.py --lang_list en --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en.log
./best_system.py --log_file ../work/mlbl_b_tanh_sigmoid/en.log
### Epoch 99 Perplexity 4.99413567038

### Test on English
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/en/99 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.32024 perplexity: 4.99413567038
# Test cost mean: 2.31623 perplexity: 4.9803009344

### Test on French
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/en/99 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 6.50365 perplexity: 90.7388081054
# Test cost mean: 6.48801 perplexity: 89.7603028281 
####################################################################################################




./train_mnlm.py --lang_list fr --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/fr.log

./best_system.py --log_file ../work/mlbl_b_tanh_sigmoid/fr.log                                                                                                 
### Epoch 94 Perplexity 5.31787918591

### Test on English
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/fr/94 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 6.80099 perplexity: 111.50715605
# Test cost mean: 6.78286 perplexity: 110.114362744

### Test on French
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/fr/94 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.41085 perplexity: 5.31787918591
# Test cost mean: 2.41261 perplexity: 5.32435745656


####################################################################################################


./train_mnlm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr.log
./best_system.py --log_file ../work/mlbl_b_tanh_sigmoid/en_fr.log 
### Epoch 98 Perplexity 5.33426200025


### Test on English
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/en_fr/98 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.37828 perplexity: 5.19915634762
# Test cost mean: 2.3779 perplexity: 5.19780241176

### Test on French
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/en_fr/98 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.43235 perplexity: 5.39772631133
# Test cost mean: 2.43588 perplexity: 5.41096051665

####################################################################################################

lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/no_lang_vector

./train_mnlm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_no_lang_vector.log
                
./best_system.py --log_file ../work/mlbl_b_tanh_sigmoid/no_lang_vector/en_fr_no_lang_vector.log 
### Epoch 100 Perplexity 5.71465938494

### Test on English
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/no_lang_vector/en_fr/100 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat.

### Output
# Dev cost mean: 2.55924 perplexity: 5.89399105573
# Test cost mean: 2.5597 perplexity: 5.89584298914

### Test on French
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/no_lang_vector/en_fr/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat.    

### Output
# Dev cost mean: 2.49406 perplexity: 5.63362531751
# Test cost mean: 2.49776 perplexity: 5.64809124403
####################################################################################################


lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/lang_id
#tmux 2
./train_mnlm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_lang_id.log
                
./best_system.py --log_file ../work/mlbl_b_tanh_sigmoid/no_lang_vector/en_fr_lang_id.log 
### Epoch 

### Test on English
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/no_lang_vector/en_fr/100 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat.

### Output
# 

### Test on French
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/no_lang_vector/en_fr/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/zero/feat.    

### Output
# 
####################################################################################################


####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
####################
lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang


./train_mplm_learn_lang.py --lang_list en --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en.log

./best_system.py --log_file  ../work/mlbl_b_learn_lang/en.log
### Epoch 98 Perplexity 4.99681622528

### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/en/98 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.32101 perplexity: 4.99681622528
# Test cost mean: 2.3171 perplexity: 4.98328123195

### Test on French
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/en/98 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 6.23251 perplexity: 75.1922463567
# Test cost mean: 6.22495 perplexity: 74.7991183345
####################################################################################################


./train_mplm_learn_lang.py --lang_list fr --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang/fr.log
### Epoch 98 Perplexity 5.31975141963

### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/fr/98 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 6.69101 perplexity: 103.322214541
# Test cost mean: 6.66932 perplexity: 101.780390047

### Test on French
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/fr/98 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.41136 perplexity: 5.31975141963
# Test cost mean: 2.41453 perplexity: 5.33144447502

####################################################################################################



./train_mplm_learn_lang.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang/en_fr.log
### Epoch 99 Perplexity 5.33365818248

### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/en_fr/99 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.37723 perplexity: 5.19537377118
# Test cost mean: 2.37703 perplexity: 5.19465003601

### Test on French
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/en_fr/99 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.4326 perplexity: 5.39866480189
# Test cost mean: 2.43482 perplexity: 5.40695593477

####################################################################################################
### + Typology language vector


network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_vector

#tmux 5
./train_mplm_learn_lang.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_plus_lang_vector.log

./best_system.py --log_file ../work/mlbl_b_learn_lang/plus_lang_vector/en_fr_plus_lang_vector.log 
### Epoch 100 Perplexity 5.33885413207 

### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_vector/en_fr/100 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.37741 perplexity: 5.19601774844
# Test cost mean: 2.37558 perplexity: 5.18943748706

### Test on French
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_vector/en_fr/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.43457 perplexity: 5.40602851042
# Test cost mean: 2.43725 perplexity: 5.41606618509

####################################################################################################
### + Typology language vector + lang_id

#tmux3

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_id

./train_mplm_learn_lang.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_plus_lang_id.log

./best_system.py --log_file ../work/mlbl_b_learn_lang/plus_lang_vector/en_fr_plus_lang_id.log 
### Epoch 

### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_vector/en_fr/?? --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output

### Test on French
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_vector/en_fr/?? --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output


















####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + ngram shape
####################
lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape
#tmux0


./train_mplm_learn_lang_shape.py --lang_list en --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en.log

./best_system.py --log_file  ../work/mlbl_b_learn_lang_shape/en.log
### 

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en/?? \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# 

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en/?? --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# 

####################################################################################################

#tmux5

./train_mplm_learn_lang_shape.py --lang_list fr --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/fr.log
###

### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/fr/?? --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# 

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/fr/?? --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# 

####################################################################################################

#tmux4

./train_mplm_learn_lang_shape.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/en_fr.log
### 

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en_fr/?? --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# 

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en_fr/?? --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# 

####################################################################################################
### + Typology language vector


network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector


./train_mplm_learn_lang_shape.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_plus_lang_vector.log

./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr_plus_lang_vector.log 
### 

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr/?? --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# 

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr/?? --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
#

####################################################################################################
### + Typology language vector + lang_id


network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_id

./train_mplm_learn_lang_shape.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_plus_lang_id.log

./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr_plus_lang_id.log 
### Epoch 

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr/?? --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr/?? --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output


