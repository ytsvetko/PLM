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

./train_mplm.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_lang_id.log

./best_system.py --log_file ../work/mlbl_b_orig/lang_id/en_fr_lang_id.log
### Epoch 78 Perplexity 5.96972064475

### Test on English
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/lang_id/en_fr/78 --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat. --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.52047 perplexity: 5.73767245781
# Test cost mean: 2.52374 perplexity: 5.75071461882


### Test on French
./test_mlbl_b_orig.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_orig/lang_id/en_fr/78 --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat. --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.60405 perplexity: 6.07990879805
# Test cost mean: 2.6065 perplexity: 6.09022638695


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
                
./best_system.py --log_file ../work/mlbl_b_tanh_sigmoid/lang_id/en_fr_lang_id.log 
### Epoch 100 Perplexity 5.34114417752  

### Test on English
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/lang_id/en_fr/100 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat. #--symbol_table /usr1/home/ytsvetko/projects/mnlm/work/symbol_table.mplm_learn_lang.en_ru_fr_ro_it_mt

### Output
# Dev cost mean: 2.37972 perplexity: 5.20434769625
# Test cost mean: 2.38068 perplexity: 5.2078355727


### Test on French
./test_mnlm.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_tanh_sigmoid/lang_id/en_fr/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr --lang_vector_path /usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat.    

### Output
# Dev cost mean: 2.43442 perplexity: 5.40545230087
# Test cost mean: 2.43703 perplexity: 5.41526338064

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

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_id

./train_mplm_learn_lang.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_plus_lang_id.log

./best_system.py --log_file ../work/mlbl_b_learn_lang/plus_lang_id/en_fr_plus_lang_id.log 
### Epoch 93 Perplexity 5.3393244158


### Test on English
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_id/en_fr/93 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
#Dev cost mean: 2.37717 perplexity: 5.19516170547
#Test cost mean: 2.3717 perplexity: 5.17550389765

### Test on French
./test_mplm_learn_lang.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/plus_lang_id/en_fr/93 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.43487 perplexity: 5.4071587742
# Test cost mean: 2.43737 perplexity: 5.4165235773



####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + ngram shape
####################
lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape

./train_mplm_learn_lang_shape.py --lang_list en --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en.log

./best_system.py --log_file  ../work/mlbl_b_learn_lang_shape/en.log
### Epoch 100 Perplexity 4.99492639594

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en/100 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.32046 perplexity: 4.99492639594
# Test cost mean: 2.31732 perplexity: 4.98407023892

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 6.48465 perplexity: 89.5515685613
# Test cost mean: 6.47613 perplexity: 89.0244682824


####################################################################################################

./train_mplm_learn_lang_shape.py --lang_list fr --batch_size 20 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/fr.log
### Epoch 97 Perplexity 5.31637835903

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/fr/97 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 6.9452 perplexity: 123.229167856
# Test cost mean: 6.92441 perplexity: 121.465854794

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/fr/97 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.41044 perplexity: 5.31637835903
# Test cost mean: 2.41335 perplexity: 5.32710168821

####################################################################################################

./train_mplm_learn_lang_shape.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/en_fr.log
### Epoch 93 Perplexity 5.33315578768

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en_fr/93 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.37645 perplexity: 5.19258669957
# Test cost mean: 2.37435 perplexity: 5.18503042699

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/en_fr/93 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.43276 perplexity: 5.39925010294
# Test cost mean: 2.43353 perplexity: 5.40213649337

####################################################################################################
### + language id

lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/lang_id/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector


./train_mplm_learn_lang_shape.py --lang_list en_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr_plus_lang_vector.log

./best_system.py --log_file ../work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr_plus_lang_vector.log 
### Epoch 99 Perplexity 5.3249734214

### Test on English
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr/99 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en

### Output
# Dev cost mean: 2.37467 perplexity: 5.18618562154
# Test cost mean: 2.37295 perplexity: 5.18000900893

### Test on French
./test_mplm_learn_lang_shape.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang_shape/plus_lang_vector/en_fr/99 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.43033 perplexity: 5.39018506334
# Test cost mean: 2.43181 perplexity: 5.39571428085

####################################################################################################









####################################################################################################
####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + input embedding matrix
####################
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings

./train_mplm_with_embeddings.py --lang_list en --batch_size 20 --save_network \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en.log

./best_system.py --log_file  ../work/mplm_with_embeddings/en.log

### Test on English
./test_mplm_with_embeddings.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings/en/97 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.31173 perplexity: 4.96479107129
# Test cost mean: 2.3073 perplexity: 4.94957080255

### Test on French
./test_mplm_with_embeddings.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings/en/97 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 6.48081 perplexity: 89.3136768936
# Test cost mean: 6.48543 perplexity: 89.6004192984
####################################################################################################

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings

./train_mplm_with_embeddings.py --lang_list fr --batch_size 20 --save_network \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/fr.log

./best_system.py --log_file  ../work/mplm_with_embeddings/fr.log
### Epoch 100 Perplexity 5.2940829447


### Test on English
# Dev cost mean: 5.76121 perplexity: 54.2370791168
# Test cost mean: 5.74588 perplexity: 53.6637273344


./test_mplm_with_embeddings.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings/fr/100 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output


### Test on French
./test_mplm_with_embeddings.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings/fr/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.40438 perplexity: 5.2940829447
# Test cost mean: 2.40665 perplexity: 5.30241500228

####################################################################################################

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings

./train_mplm_with_embeddings.py --lang_list en_fr --batch_size 20 --save_network \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/en_fr.log

./best_system.py --log_file  ../work/mplm_with_embeddings/en_fr.log
### Epoch 99 Perplexity 5.2961279709

### Test on English
./test_mplm_with_embeddings.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings/en_fr/99\
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.36609 perplexity: 5.15543311531
# Test cost mean: 2.36421 perplexity: 5.14869578333


### Test on French
./test_mplm_with_embeddings.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_embeddings/en_fr/99 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.42285 perplexity: 5.36231102259
# Test cost mean: 2.42657 perplexity: 5.37613535688



















####################################################################################################
####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + attention model
####################
# alpha = 0.9, betta = 0.1

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1

./train_mplm_with_attention.py --lang_list en --batch_size 20 --save_network \
                --network_dir ${network_dir} --alpha 0.9 --betta 0.1  2>&1 | tee ${network_dir}/en.log

./best_system.py --log_file  ../work/mplm_with_attention_alpha_0.9_betta_0.1/en.log
### Epoch 100 Perplexity 4.96578723058


### Test on English
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/en/100 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.31202 perplexity: 4.96578723058
# Test cost mean: 2.30909 perplexity: 4.95571259641

### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/en/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 6.5486 perplexity: 93.6105510493
# Test cost mean: 6.55102 perplexity: 93.7680442426

####################################################################################################
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1

./train_mplm_with_attention.py --lang_list fr --batch_size 20 --save_network \
                --network_dir ${network_dir}  --alpha 0.9 --betta 0.1   2>&1 | tee ${network_dir}/fr.log

./best_system.py --log_file  ../work/mplm_with_attention_alpha_0.9_betta_0.1/fr.log
### Epoch 95 Perplexity 5.2921217918

### Test on English

./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/fr/95 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 5.90859 perplexity: 60.0707310739
# Test cost mean: 5.89987 perplexity: 59.708648615


### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/fr/95 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.40385 perplexity: 5.2921217918
# Test cost mean: 2.40642 perplexity: 5.30158786539


####################################################################################################
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1

./train_mplm_with_attention.py --lang_list en_fr --batch_size 20 --save_network \
                --network_dir ${network_dir}  --alpha 0.9 --betta 0.1  2>&1 | tee ${network_dir}/en_fr.log

./best_system.py --log_file  ../work/mplm_with_attention_alpha_0.9_betta_0.1/en_fr.log
### Epoch 98 Perplexity 5.2949762886

### Test on English
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/en_fr/98\
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.36409 perplexity: 5.14828993474
# Test cost mean: 2.36354 perplexity: 5.14630114366

### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/en_fr/98 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.42332 perplexity: 5.36402781067
# Test cost mean: 2.42602 perplexity: 5.37408963285



####################################################################################################



####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + attention model
####################
# alpha = 0.7, betta = 0.3

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3

./train_mplm_with_attention.py --lang_list en --batch_size 20 --save_network \
                --network_dir ${network_dir} --alpha 0.7 --betta 0.3  2>&1 | tee ${network_dir}/en.log

./best_system.py --log_file  ../work/mplm_with_attention_alpha_0.7_betta_0.3/en.log
### Epoch 100 Perplexity 4.96285430242


### Test on English
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3/en/100 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.31117 perplexity: 4.96285430242
# Test cost mean: 2.30819 perplexity: 4.95263010733

### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3/en/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 6.85393 perplexity: 115.674430173
# Test cost mean: 6.85563 perplexity: 115.811421881

####################################################################################################
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3


./train_mplm_with_attention.py --lang_list fr --batch_size 20 --save_network \
                --network_dir ${network_dir}  --alpha 0.7 --betta 0.3   2>&1 | tee ${network_dir}/fr.log

./best_system.py --log_file  ../work/mplm_with_attention_alpha_0.7_betta_0.3/fr.log
###Epoch 93 Perplexity 5.28656855388


### Test on English

./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/fr/93 \
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 6.09041 perplexity: 68.1390193274
# Test cost mean: 6.07929 perplexity: 67.615709797


### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.9_betta_0.1/fr/93 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.40233 perplexity: 5.28656855388
# Test cost mean: 2.40498 perplexity: 5.29629164213

####################################################################################################
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3

./train_mplm_with_attention.py --lang_list en_fr --batch_size 20 --save_network \
                --network_dir ${network_dir}  --alpha 0.7 --betta 0.3  2>&1 | tee ${network_dir}/en_fr.log

0


./best_system.py --log_file  ../work/mplm_with_attention_alpha_0.7_betta_0.3/en_fr.log
###Epoch 100 Perplexity 5.29937608352

### Test on English
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3/en_fr/100\
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en
### Output
# Dev cost mean: 2.36524 perplexity: 5.15238902865
# Test cost mean: 2.36439 perplexity: 5.14935865308

### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_0.7_betta_0.3/en_fr/100 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr

### Output
# Dev cost mean: 2.42454 perplexity: 5.36856571811
# Test cost mean: 2.42813 perplexity: 5.38195789208
####################################################################################################
# alpha = 1, betta = 0

network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_1_betta_0

./train_mplm_with_attention.py --lang_list en_fr --batch_size 20 --save_network \
                --network_dir ${network_dir} --alpha 1.0 --betta 0.0 2>&1 | tee ${network_dir}/en_fr.log

./best_system.py --log_file  ../work/mplm_with_attention_alpha_1_betta_0/en_fr.log
### Epoch 96 Perplexity 4.2334016959

### Test on English
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_1_betta_0/en_fr/96\
                 --lang_list en --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.en \
                 --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.en  --alpha 1.0 --betta 0.0 
### Output
#Dev cost mean: 2.36255 perplexity: 5.1427983857
#Test cost mean: 2.36359 perplexity: 5.14650611184


### Test on French
./test_mplm_with_attention.py --network_dir /usr1/home/ytsvetko/projects/mnlm/work/mplm_with_attention_alpha_1_betta_0/en_fr/96 --lang_list fr --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.fr --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.fr  --alpha 1.0 --betta 0.0 

### Output
# Dev cost mean: 2.42112 perplexity: 5.35586800209
# Test cost mean: 2.42387 perplexity: 5.36609187438

####################################################################################################






###################################################################################################
####################################################################################################
# LOANWORDS1
lang_vector_path=/usr1/home/ytsvetko/projects/mnlm/data/wals/feat.
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang

./train_mplm_learn_lang.py --lang_list ro_fr --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/ro_fr.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang/ro_fr.log

./generate_word_vectors.py --pron_dict /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.ro --out_word_vectors /usr1/home/ytsvetko/projects/mnlm/data/loanwords/ro-fr/pron-dict.vectors.ro --in_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/ro_fr/92/vectors --in_softmax_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/ro_fr/92/softmax_vectors

./generate_word_vectors.py --pron_dict /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.fr --out_word_vectors /usr1/home/ytsvetko/projects/mnlm/data/loanwords/ro-fr/pron-dict.vectors.fr --in_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/ro_fr/92/vectors --in_softmax_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/ro_fr/92/softmax_vectors

./loanwords.py  --src_tgt_pairs ../data/loanwords/ro-fr/test.en-ro-fr --src_vectors ../data/loanwords/ro-fr/pron-dict.vectors.ro --tgt_vectors ../data/loanwords/ro-fr/pron-dict.vectors.fr --num_closest 10 --out_filename ../data/loanwords/ro-fr/mlbl_b_learn_lang.phone-levenshtein 2>&1 | tee ../data/loanwords/ro-fr/ro-fr.loanwords.log

#############################################
./train_mplm_learn_lang.py --lang_list mt_it --batch_size 40 --save_network \
                --lang_vector_path ${lang_vector_path} \
                --network_dir ${network_dir}  2>&1 | tee ${network_dir}/mt_it.log
                
./best_system.py --log_file ../work/mlbl_b_learn_lang/mt_it.log
# Epoch 100 Perplexity 4.28188938896

./generate_word_vectors.py  --pron_dict /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.mt --out_word_vectors /usr1/home/ytsvetko/projects/mnlm/data/loanwords/pron-dict.vectors.mt --in_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/mt_it/100/vectors --in_softmax_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/mt_it/100/softmax_vectors

./generate_word_vectors.py  --pron_dict /usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.it --out_word_vectors /usr1/home/ytsvetko/projects/mnlm/data/loanwords/pron-dict.vectors.it --in_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/mt_it/100/vectors --in_softmax_vectors /usr1/home/ytsvetko/projects/mnlm/work/mlbl_b_learn_lang/mt_it/100/softmax_vectors


./loanwords.py  --src_tgt_pairs ../data/loanwords/mt-it/test.en-mt-it --src_vectors ../data/loanwords/mt-it/pron-dict.vectors.mt --tgt_vectors ../data/loanwords/mt-it/pron-dict.vectors.it --num_closest 10 --out_filename ../data/loanwords/mt-it/mlbl_b_learn_lang.phone-levenshtein 2>&1 | tee ../data/loanwords/mt-it/mt-it.loanwords.log
# Vectors: Total: 27, Correct: 17, Accuracy: 0.6296296296296297  
# Softmax vectors: Total: 27, Correct: 10, Accuracy: 0.37037037037037035  



####################################################################################################
####################################################################################################




























####################################################################################################
####################################################################################################

####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + append language vector to context layer
# + context = 2 on left, 2 on right
####################
### English
# tmux
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context
lang=en

./train_mplm_context.py --lang_list ${lang} --batch_size 20 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### Epoch 93 Perplexity 3.18260286183


### Test on English
./test_mplm_context.py --network_dir ${network_dir}/${lang}/93 \
  --lang_list ${lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${lang}
### Output
# Dev cost mean: 1.67021 perplexity: 3.18260286183
# Test cost mean: 1.66054 perplexity: 3.16135742628


####################################################################################################
### Swahili--Arabic
# tmux
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context
src_lang=sw
tgt_lang=ar
lang=${src_lang}_${tgt_lang}

./train_mplm_context.py --lang_list ${lang} --batch_size 40 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### Epoch 14 Perplexity 3.16638235683 ### KILLED 


### Test on Swahili
./test_mplm_context.py --network_dir ${network_dir}/${lang}/14 \
  --lang_list ${src_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${src_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${src_lang}
### Output
# Dev cost mean: 1.47971 perplexity: 2.78892502048
# Test cost mean: 1.48581 perplexity: 2.80074338281


### Test on Arabic
./test_mplm_context.py --network_dir ${network_dir}/${lang}/14 \
  --lang_list ${tgt_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${tgt_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${tgt_lang}
### Output
# Dev cost mean: 1.73545 perplexity: 3.32984365406
# Test cost mean: 1.73589 perplexity: 3.33084340078

####################################################################################################
### Romanian--French
# tmux
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context
src_lang=ro
tgt_lang=fr
lang=${src_lang}_${tgt_lang}

./train_mplm_context.py --lang_list ${lang} --batch_size 40 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### Epoch 58 Perplexity 2.63213143594  ### KILLED 

### Test on Romanian
./test_mplm_context.py --network_dir ${network_dir}/${lang}/58 \
  --lang_list ${src_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${src_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${src_lang}
### Output
# Dev cost mean: 1.12342 perplexity: 2.17862926478
# Test cost mean: 1.12076 perplexity: 2.17460811044

### Test on French
./test_mplm_context.py --network_dir ${network_dir}/${lang}/58 \
  --lang_list ${tgt_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${tgt_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${tgt_lang}
### Output
# Dev cost mean: 2.12155 perplexity: 4.35162613108
# Test cost mean: 2.12867 perplexity: 4.37313555866


####################################################################################################
### Maltese--Italian
# tmux
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context
src_lang=mt
tgt_lang=it
lang=${src_lang}_${tgt_lang}

./train_mplm_context.py --lang_list ${lang} --batch_size 40 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### Epoch 96 Perplexity 2.94116536554


### Test on Maltese
./test_mplm_context.py --network_dir ${network_dir}/${lang}/96 \
  --lang_list ${src_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${src_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${src_lang}
### Output
# Dev cost mean: 1.50448 perplexity: 2.83723257166
# Test cost mean: 1.51315 perplexity: 2.85431699365

### Test on Italian
./test_mplm_context.py --network_dir ${network_dir}/${lang}/100 \
  --lang_list ${tgt_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${tgt_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${tgt_lang}
### Output
# Dev cost mean: 1.62482 perplexity: 3.08404269062
# Test cost mean: 1.62846 perplexity: 3.09182311597


####################################################################################################






















####################################################################################################
####################################################################################################

####################
# Modality-Biased Log-Bilinear Model 
# + tanh activation
# + learn language vector
# + append language vector to context layer
# + context = 2 on left, 2 on right
# + INS symbol in the middle
####################
### English
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords
lang=en

./train_mplm_context.py --lang_list ${lang} --batch_size 20 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### Epoch 97 Perplexity 3.18239643146

### Test on English
./test_mplm_context.py --network_dir ${network_dir}/${lang}/97 \
  --lang_list ${lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${lang}
### Output
# Dev cost mean: 1.67011 perplexity: 3.18239643146
# Test cost mean: 1.66123 perplexity: 3.16285144456


####################################################################################################
### Swahili--Arabic
# tmux2
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords
src_lang=sw
tgt_lang=ar
lang=${src_lang}_${tgt_lang}

./train_mplm_context.py --lang_list ${lang} --batch_size 40 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### 


### Test on Swahili
./test_mplm_context.py --network_dir ${network_dir}/${lang}/? \
  --lang_list ${src_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${src_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${src_lang}
### Output
# 
# 


### Test on Arabic
./test_mplm_context.py --network_dir ${network_dir}/${lang}/? \
  --lang_list ${tgt_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${tgt_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${tgt_lang}
### Output
# 

####################################################################################################
### Romanian--French
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords
src_lang=ro
tgt_lang=fr
lang=${src_lang}_${tgt_lang}

./train_mplm_context.py --lang_list ${lang} --batch_size 40 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### Epoch 99 Perplexity 2.6179886597

### Test on Romanian
./test_mplm_context.py --network_dir ${network_dir}/${lang}/99 \
  --lang_list ${src_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${src_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${src_lang}
### Output
# Dev cost mean: 1.11468 perplexity: 2.16547516694
# Test cost mean: 1.11249 perplexity: 2.16218352739

### Test on French
./test_mplm_context.py --network_dir ${network_dir}/${lang}/99 \
  --lang_list ${tgt_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${tgt_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${tgt_lang}
### Output
# Dev cost mean: 2.11637 perplexity: 4.33602209569
# Test cost mean: 2.11914 perplexity: 4.34435301897

####################################################################################################
### Maltese--Italian
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords
src_lang=mt
tgt_lang=it
lang=${src_lang}_${tgt_lang}

./train_mplm_context.py --lang_list ${lang} --batch_size 40 --save_network \
                --network_dir ${network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### 


### Test on Maltese
./test_mplm_context.py --network_dir ${network_dir}/${lang}/100 \
  --lang_list ${src_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${src_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${src_lang}
### Output
# Dev cost mean: 1.50468 perplexity: 2.83761145093
# Test cost mean: 1.51421 perplexity: 2.85641754785


### Test on Italian
./test_mplm_context.py --network_dir ${network_dir}/${lang}/100 \
  --lang_list ${tgt_lang} --dev_path /usr1/home/ytsvetko/projects/mnlm/data/pron/dev/pron-dict.${tgt_lang} \
  --test_path /usr1/home/ytsvetko/projects/mnlm/data/pron/test/pron-dict.${tgt_lang}
### Output
#Dev cost mean: 1.62011 perplexity: 3.07399014177
# Test cost mean: 1.62617 perplexity: 3.08692237478



####################################################################################################
















####################################################################################################

# ####################
# LOANWORDS Scoring Model 
# ####################

####################################################################################################
### Swahili--Arabic
# tmux6, tmux1
src_lang=sw
tgt_lang=ar
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_loanwords
lang=${src_lang}_${tgt_lang}
init_network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords/${lang}/46
loanwords_training=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/train.en-${src_lang}-${tgt_lang}
loanwords_test=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/test.en-${src_lang}-${tgt_lang}
src_pronunciations=/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.${src_lang}
tgt_pronunciations=/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.${tgt_lang}
test_output_file=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/${lang}_nbest.out

./train_mplm_loanwords.py --loanwords_training ${loanwords_training} \
       --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
       --src_pronunciations ${src_pronunciations} --tgt_pronunciations ${tgt_pronunciations} \
       --batch_size 40 --save_network --network_dir ${network_dir} \
       --load_network --load_network_dir ${init_network_dir} 2>&1 | tee ${network_dir}/${lang}.log

./best_system.py --log_file  ${network_dir}/${lang}.log
### 


### Test on Swahili
./test_mplm_loanwords.py --loanwords_test ${loanwords_test} \
       --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
       --load_network_dir ${network_dir}/${lang}/99 \
       --output_file ${test_output_file}
### Output
# 
#Gahl!! {:
####################################################################################################
### Romanian--French
# tmux3
src_lang=ro
tgt_lang=fr
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_loanwords
lang=${src_lang}_${tgt_lang}
init_network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords/${lang}/99
loanwords_training=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/train.en-${src_lang}-${tgt_lang}
loanwords_test=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/test.en-${src_lang}-${tgt_lang}
src_pronunciations=/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.${src_lang}
tgt_pronunciations=/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.${tgt_lang}
test_output_file=${network_dir}/${lang}_nbest.out

./train_mplm_loanwords.py --loanwords_training ${loanwords_training} \
       --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
       --src_pronunciations ${src_pronunciations} --tgt_pronunciations ${tgt_pronunciations} \
       --batch_size 40 --save_network --network_dir ${network_dir} \
       --load_network --load_network_dir ${init_network_dir} 2>&1 | tee ${network_dir}/${lang}.log


./best_system.py --log_file  ${network_dir}/${lang}.log
### 

### Test on Romanian
./test_mplm_loanwords.py --loanwords_test ${loanwords_test} \
       --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
       --load_network_dir ${network_dir}/${lang}/50 \
       --output_file ${test_output_file}


####################################################################################################
### Maltese--Italian
# tmux5
src_lang=mt
tgt_lang=it
network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_loanwords
lang=${src_lang}_${tgt_lang}
init_network_dir=/usr1/home/ytsvetko/projects/mnlm/work/mplm_context_loanwords/${lang}/99
loanwords_training=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/train.en-${src_lang}-${tgt_lang}
loanwords_test=/usr1/home/ytsvetko/projects/mnlm/data/loanwords/${src_lang}-${tgt_lang}/test.en-${src_lang}-${tgt_lang}
src_pronunciations=/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.${src_lang}
tgt_pronunciations=/usr1/home/ytsvetko/projects/mnlm/data/pron/pron-dict.${tgt_lang}
test_output_file=${network_dir}/${lang}_nbest.out

./train_mplm_loanwords.py --loanwords_training ${loanwords_training} \
       --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
       --src_pronunciations ${src_pronunciations} --tgt_pronunciations ${tgt_pronunciations} \
       --batch_size 40 --save_network --network_dir ${network_dir} \
       --load_network --load_network_dir ${init_network_dir} 2>&1 | tee ${network_dir}/${lang}.log


./best_system.py --log_file  ${network_dir}/${lang}.log
### 


### Test on Maltese
./test_mplm_loanwords.py --loanwords_test ${loanwords_test} \
       --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
       --load_network_dir ${network_dir}/${lang}/49 \
       --output_file ${test_output_file}


####################################################################################################






