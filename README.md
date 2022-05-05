# 1. Reference

[GitHub - xing-hu/EMSE-DeepCom: The dataset for EMSE-DeepCom](https://github.com/xing-hu/EMSE-DeepCom)

[https://xin-xia.github.io/publication/emse192.pdf](https://xin-xia.github.io/publication/emse192.pdf)

# 2. Project Structure

- config
- data_utils
- dataset
- emse-data
    
    
    | folder | data |
    | --- | --- |
    | model/hybrid | checkpoints, data, eval, config.yaml, default.yaml, log.txt |
    | train | train_ast.json, train.token.ast, train.token.code, train.token.nl |
    | test | test_ast.json, test.token.ast, test.token.code, test.token.nl |
    | valid | valid_ast.json, valid.token.ast, valid.token.code, valid.token.nl |
    | vocab... | vocab.ast, vocab.code, vocab.nl |
    
    ### Training/Test/Valid Data
    
    - code
    
    ```java
    public int entrySize(Object key, Object value) throws lllegalArgumentException{
    	if (value==Token . TOMBSTONE) {
    		return NUM_;}
    	int size = HeapLRUCapacityController .this .getPerEntryOverhead();
    	size += sizeof (key);
    	size += sizeof(value);
    	return size;
    }
    
    >>====================================================================================================================================================================================================================================================================================
    public int entrySize ( Object key , Object value ) throws IllegalArgumentException { if ( value == Token . TOMBSTONE ) { return NUM_ ; } int size = HeapLRUCapacityController . this . getPerEntryOverhead ( ) ; size += sizeof ( key ) ; size += sizeof ( value ) ; return size ; }
    >>====================================================================================================================================================================================================================================================================================
    ```
    
    - ast.json
    
    ```java
    [{"id": 0, "type": "MethodDeclaration", "children": [1, 2, 4, 6, 13, 18, 23, 28], "value": "entrySize"}, {"id": 1, "type": "BasicType", "value": "int"}, {"id": 2, "type": "FormalParameter", "children": [3], "value": "key"}, {"id": 3, "type": "ReferenceType", "value": "Object"}, {"id": 4, "type": "FormalParameter", "children": [5], "value": "value"}, {"id": 5, "type": "ReferenceType", "value": "Object"}, {"id": 6, "type": "IfStatement", "children": [7, 10]}, {"id": 7, "type": "BinaryOperation", "children": [8, 9]}, {"id": 8, "type": "MemberReference", "value": "value"}, {"id": 9, "type": "MemberReference", "value": "Token.TOMBSTONE"}, {"id": 10, "type": "BlockStatement", "children": [11], "value": "None"}, {"id": 11, "type": "ReturnStatement", "children": [12], "value": "return"}, {"id": 12, "type": "MemberReference", "value": "NUM_"}, {"id": 13, "type": "LocalVariableDeclaration", "children": [14, 15], "value": "int"}, {"id": 14, "type": "BasicType", "value": "int"}, {"id": 15, "type": "VariableDeclarator", "children": [16], "value": "size"}, {"id": 16, "type": "This", "children": [17], "value": "HeapLRUCapacityController.this.getPerEntryOverhead"}, {"id": 17, "type": "MethodInvocation", "value": "."}, {"id": 18, "type": "StatementExpression", "children": [19]}, {"id": 19, "type": "Assignment", "children": [20, 21]}, {"id": 20, "type": "MemberReference", "value": "size"}, {"id": 21, "type": "MethodInvocation", "children": [22], "value": "sizeof"}, {"id": 22, "type": "MemberReference", "value": "key"}, {"id": 23, "type": "StatementExpression", "children": [24]}, {"id": 24, "type": "Assignment", "children": [25, 26]}, {"id": 25, "type": "MemberReference", "value": "size"}, {"id": 26, "type": "MethodInvocation", "children": [27], "value": "sizeof"}, {"id": 27, "type": "MemberReference", "value": "value"}, {"id": 28, "type": "ReturnStatement", "children": [29], "value": "return"}, {"id": 29, "type": "MemberReference", "value": "size"}]
    ```
    
    - token.ast
    
    ```java
    ( MethodDeclaration ( BasicType ) BasicType ( FormalParameter ( ReferenceType ) ReferenceType ) FormalParameter ( FormalParameter ( ReferenceType ) ReferenceType ) FormalParameter ( IfStatement ( BinaryOperation ( MemberReference ) MemberReference ( MemberReference ) MemberReference ) BinaryOperation ( BlockStatement ( ReturnStatement ( MemberReference ) MemberReference ) ReturnStatement ) BlockStatement ) IfStatement ( LocalVariableDeclaration ( BasicType ) BasicType ( VariableDeclarator ( This ( MethodInvocation ) MethodInvocation ) This ) VariableDeclarator ) LocalVariableDeclaration ( StatementExpression ( Assignment ( MemberReference ) MemberReference ( MethodInvocation ( MemberReference ) MemberReference ) MethodInvocation ) Assignment ) StatementExpression ( StatementExpression ( Assignment ( MemberReference ) MemberReference ( MethodInvocation ( MemberReference ) MemberReference ) MethodInvocation ) Assignment ) StatementExpression ( ReturnStatement ( MemberReference ) MemberReference ) ReturnStatement ) MethodDeclaration
    ```
    
    - token.nl
    
    ```java
    "as far as we re concerned all entries have the same size"
    ```
    
- img
- projects
- scripts
- source code
    - __ main __ .py
    - beam_search.py
    - evaluations.py
    - models.py
    - rnn.py
    - seq2seq_model.py
    - translation_model.py
    - utils.py

# 3. Execute File

### 🌼 Model Training

- python3 __**main**__.py config.yaml --train -v
- config.yaml 에서 hyper parameter 수정 가능
- 적절하게 수행

```java
parser.add_argument('config', help='load a configuration file in the YAML format')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
parser.add_argument('--debug', action='store_true', help='debug mode')

# using 'store_const' instead of 'store_true' so that the default value is `None` instead of `False`
parser.add_argument('--reset', action='store_const', const=True, help="reset model (don't load any checkpoint)")
parser.add_argument('--reset-learning-rate', action='store_const', const=True, help='reset learning rate')
parser.add_argument('--learning-rate', type=float, help='custom learning rate (triggers `reset-learning-rate`)')
parser.add_argument('--purge', action='store_true', help='remove previous model files')
```

### 🌼 Code Comment Generation Test

- python3 __**main**__.py config.yaml --decode *“data_dir”*
- data_dir : ast.json, token.ast, token.code, token.nl 있는 폴더

### 🌼 Automatic Comment Evaluation

- python3 __**main**__.py config.yaml --eval *“data_dir”*
- data_dir : ast.json, token.ast, token.code, token.nl 있는 폴더

### 🌼 ****Generate ASTs for Java method****

- python3 get_ast.py source.code ast.json

```java
// code
public boolean doesNotHaveIds (){ 
  return getIds () == null || getIds ().getIds().isEmpty(); 
}
```

```java
// AST
[
{"id": 0, "type": "MethodDeclaration", "children": [1, 2], "value": "doesNotHaveIds"}, 
    {"id": 1, "type": "BasicType", "value": "boolean"}, 
    {"id": 2, "type": "ReturnStatement", "children": [3], "value": "return"}, 
        {"id": 3, "type": "BinaryOperation", "children": [4, 7]}, 
            {"id": 4, "type": "BinaryOperation", "children": [5, 6]}, 
                {"id": 5, "type": "MethodInvocation", "value": "getIds"}, 
                {"id": 6, "type": "Literal", "value": "null"}, 
            {"id": 7, "type": "MethodInvocation", "children": [8, 9], "value": "getIds"}, 
                {"id": 8, "type": "MethodInvocation", "value": "."}, 
                {"id": 9, "type": "MethodInvocation", "value": "."}
 ]
```

### 🌼 AST_Traversal(Generate SBT)

- python3 ast_traversal.py
- def get_sbt_structrue

```java
// SBT
( MethodDeclaration ( BasicType ) BasicType ( FormalParameter ( ReferenceType ) ReferenceType ) FormalParameter ( FormalParameter ( ReferenceType ) ReferenceType ) FormalParameter ( IfStatement ( BinaryOperation ( MemberReference ) MemberReference ( MemberReference ) MemberReference ) BinaryOperation ( BlockStatement ( ReturnStatement ( MemberReference ) MemberReference ) ReturnStatement ) BlockStatement ) IfStatement ( LocalVariableDeclaration ( BasicType ) BasicType ( VariableDeclarator ( This ( MethodInvocation ) MethodInvocation ) This ) VariableDeclarator ) LocalVariableDeclaration ( StatementExpression ( Assignment ( MemberReference ) MemberReference ( MethodInvocation ( MemberReference ) MemberReference ) MethodInvocation ) Assignment ) StatementExpression ( StatementExpression ( Assignment ( MemberReference ) MemberReference ( MethodInvocation ( MemberReference ) MemberReference ) MethodInvocation ) Assignment ) StatementExpression ( ReturnStatement ( MemberReference ) MemberReference ) ReturnStatement ) MethodDeclaration
```

### 🌼 AST_Traversal(Generate SBT with Code)

- python3 ast_traversal.py
- def get_sbtcode_structure

```java
// SBT + CODE
( MethodDeclaration entrySize ( BasicType int ) BasicType ( FormalParameter key ( ReferenceType Object ) ReferenceType ) FormalParameter ( FormalParameter value ( ReferenceType Object ) ReferenceType ) FormalParameter ( IfStatement if ( BinaryOperation ( MemberReference value ) MemberReference ( MemberReference Token.TOMBSTONE ) MemberReference ) BinaryOperation ( BlockStatement { ( ReturnStatement return ( MemberReference NUM_ ) MemberReference ) ReturnStatement ) BlockStatement ) IfStatement ( LocalVariableDeclaration int ( BasicType int ) BasicType ( VariableDeclarator size ( This HeapLRUCapacityController.this.getPerEntryOverhead ( MethodInvocation . ) MethodInvocation ) This ) VariableDeclarator ) LocalVariableDeclaration ( StatementExpression size ( Assignment ( MemberReference size ) MemberReference ( MethodInvocation sizeof ( MemberReference key ) MemberReference ) MethodInvocation ) Assignment ) StatementExpression ( StatementExpression size ( Assignment ( MemberReference size ) MemberReference ( MethodInvocation sizeof ( MemberReference value ) MemberReference ) MethodInvocation ) Assignment ) StatementExpression ( ReturnStatement return ( MemberReference size ) MemberReference ) ReturnStatement ) MethodDeclaration
```

# 4. Our Experiments

### 🦑 DeepCom(default)

- config_default.yaml
    
    ```yaml
    # SGD parameters
    learning_rate: 0.5        
    sgd_learning_rate: 1.0      
    learning_rate_decay_factor: 0.99  
    
    # training parameters
    max_gradient_norm: 5.0   
    steps_per_checkpoint: 2000   
    steps_per_eval: 2000    
    eval_burn_in: 0          
    max_steps: 0             
    max_epochs: 50            
    keep_best: 5             
    feed_previous: 0.0       
    optimizer: sgd       
    moving_average: null
    
    # batch iteration parameters
    batch_size: 100         
    batch_mode: random   
    shuffle: True       
    read_ahead: 1      
    reverse_input: True
    
    # model (each one of these settings can be defined specifically in 'encoders' and 'decoders', or generally here)
    cell_size: 512          
    embedding_size: 512     
    attn_size: 256           
    layers: 1                
    cell_type: LSTM          
    character_level: False   
    truncate_lines: True
    
    # data
    max_train_size: 0        
    max_dev_size: 0          
    max_test_size: 0         
    data_dir: ../emse-data(ast_only)
    model_dir: ../emse-data(ast_only)/model/default
    train_prefix: train      
    script_dir: scripts      
    dev_prefix: test        
    vocab_prefix: vocab      
    checkpoints: []
    
    # decoding
    score_function: nltk_sentence_bleu
    post_process_script: null 
    remove_unk: False        
    beam_size: 1
    
    # general
    **encoders:                
      - name: ast            
        max_len: 500         
        attention_type: global
    
    decoders:                
      - name: nl            
        max_len: 30**
    ```
    
- 성능 확인
    
    ```yaml
    # 우리가 테스트 한 것
    step 222000 **epoch 50** learning rate 0.306 step-time 0.791 loss 10.135
    test eval: loss 36.39
    starting decoding
    test avg_score=0.2019(**BLEU**)
    
    test 폴더 - avg_score: 0.2026
    
    # 논문 결과
    BLEU: 38.17
    ```
    
    

### 🦑 Hybrid - DeepCom(default)

- config.yaml
    
    ```yaml
    # SGD parameters
    learning_rate: 0.5        # initial learning rate
    sgd_learning_rate: 1.0      # SGD can start at a different learning rate (useful for switching between Adam and SGD)
    learning_rate_decay_factor: 0.99  # decay the learning rate by this factor at a given frequency
    
    # training parameters
    max_gradient_norm: 5.0   # clip gradients to this norm (prevents exploding gradient)
    steps_per_checkpoint: 2000   # number of SGD updates between each checkpoint
    steps_per_eval: 2000    # number of SGD updates between each BLEU eval (on dev set)
    eval_burn_in: 0          # minimum number of updates before starting BLEU eval
    max_steps: 0             # maximum number of updates before stopping, 600000->0
    max_epochs: 50            # maximum number of epochs before stopping, 100->50
    keep_best: 5             # number of best checkpoints to keep (based on BLEU score on dev set)
    feed_previous: 0.0       # randomly feed prev output instead of ground truth to decoder during training ([0,1] proba)
    optimizer: sgd          # which training algorithm to use ('sgd', 'adadelta', or 'adam')
    moving_average: null     # TODO
    
    # batch iteration parameters
    batch_size: 128           # batch size (during training and greedy decoding), 64->128
    batch_mode: standard     # standard (cycle through train set) or random (sample from train set)
    shuffle: True            # shuffle dataset at each new epoch
    read_ahead: 1           # number of batches to read ahead and sort by sequence length (can speed up training)
    reverse_input: True     # reverse input sequences
    
    # model (each one of these settings can be defined specifically in 'encoders' and 'decoders', or generally here)
    cell_size: 256          # size of the RNN cells
    embedding_size: 256     # size of the embeddings
    attn_size: 128           # size of the attention layer
    layers: 1                # number of RNN layers per encoder and decoder
    cell_type: GRU          # LSTM, GRU, DropoutGRU
    character_level: False   # character-level sequences
    truncate_lines: True     # if True truncate lines which are too long, otherwise just drop them
    
    # encoder settings
    bidir: False              # use bidirectional encoders
    train_initial_states: True  # whether the initial states of the encoder should be trainable parameters
    bidir_projection: False  # project bidirectional encoder states to cell_size (or just keep the concatenation)
    time_pooling: null       # perform time pooling (skip states) between the layers of the encoder (list of layers - 1 ratios)
    pooling_avg: True        # average or skip consecutive states
    binary: False            # use binary input for the encoder (no vocab and no embeddings, see utils.read_binary_features)
    attn_filters: 0
    attn_filter_length: 0
    input_layers: null       # list of fully connected layer sizes, applied before the encoder
    attn_temperature: 1.0    # 1.0: true softmax (low values: uniform distribution, high values: argmax)
    final_state: last        # last (default), concat_last, average
    highway_layers: 0        # number of highway layers before the encoder (after convolutions and maxout)
    
    # decoder settings
    tie_embeddings: False     # use transpose of the embedding matrix for output projection (requires 'output_extra_proj')
    use_previous_word: True   # use previous word when predicting a new word
    attn_prev_word: False     # use the previous word in the attention model
    softmax_temperature: 1.0  # TODO: temperature of the output softmax
    pred_edits: False         # output is a sequence of edits, apply those edits before decoding/evaluating
    conditional_rnn: False    # two-layer decoder, where the 1st layer is used for attention, and the 2nd layer for prediction
    generate_first: True      # generate next word before updating state (look->generate->update)
    update_first: False       # update state before looking and generating next word
    rnn_feed_attn: True       # feed attention context to the RNN's transition fonction
    use_lstm_full_state: False # use LSTM's full state for attention and next word prediction
    pred_embed_proj: True     # project decoder output to embedding size before projecting to vocab size
    pred_deep_layer: False    # add a non-linear transformation just before softmax
    pred_maxout_layer: True   # use a maxout layer just before the vocabulary projection and softmax
    aggregation_method: sum # how to combine the attention contexts of multiple encoders (concat, sum)
    
    # data
    max_train_size: 0        # maximum size of the training data (0 for unlimited)
    max_dev_size: 0          # maximum size of the dev data
    max_test_size: 0         # maximum size of the test data
    data_dir: ../emse-data(original)           # directory containing the training data
    model_dir: ../emse-data(original)/model/hybrid
    train_prefix: train      # name of the training corpus
    script_dir: scripts      # directory where the scripts are kepts (in particular the scoring scripts)
    dev_prefix: test        # names of the development corpora
    vocab_prefix: vocab      # name of the vocabulary files
    checkpoints: []          # list of checkpoints to load (in this specific order) after main checkpoint
    
    # decoding
    score_function: nltk_sentence_bleu # name of the main scoring function, inside 'evaluation.py' (used for selecting models)
    post_process_script: null # path to post-processing script (called before evaluating)
    remove_unk: False        # remove UNK symbols from the decoder output
    beam_size: 5             # beam size for decoding (decoder is greedy by default)
    ensemble: False          # use an ensemble of models while decoding (specified by the --checkpoints parameter)
    output: null             # output file for decoding (writes to standard output by default)
    len_normalization: 1.0   # length normalization coefficient used in beam-search decoder
    early_stopping: True     # reduce beam-size each time a finished hypothesis is encountered (affects decoding speed)
    raw_output: False        # output translation hypotheses without any post-processing
    average: False           # like ensemble, but instead of averaging the log-probs, average all parameters
    pred_edits: False
    
    # general
    **encoders:                # this is a list (you can specify several encoders)
      - name: code             # each encoder or decoder has a name (used for naming variables) and an extension (for files)
        max_len: 200          # max_len of api
        attention_type: global
      - name: ast
        max_len: 500
        attention_type: global
    
    decoders:                # Each encoder or decoder can redefine its own values for a number of parameters,
      - name: nl             # including `cell_size`, `embedding_size` and `attn_size`
        max_len: 30**
    ```
    
- 성능 확인
    
    ```yaml
    # 우리가 테스트 한 것
    step 174000 **epoch 50** learning rate 0.306 step-time 0.951 loss 8.083
    test eval: loss 33.39
    test avg_score=0.3806(**BLEU**)
    
    test 폴더 - avg_score:0.3820

    ```
    
    

### 🦑 DeepCom(our model)

- config_park.yaml
    
    ```yaml
    # 위 항목 deepcom(default)와 동일
    # data
    max_train_size: 0        # maximum size of the training data (0 for unlimited)
    max_dev_size: 0          # maximum size of the dev data
    max_test_size: 0         # maximum size of the test data
    data_dir: ../emse-data(sbt_code)           # directory containing the training data
    model_dir: ../emse-data(sbt_code)/model/default
    train_prefix: train      # name of the training corpus
    script_dir: scripts      # directory where the scripts are kepts (in particular the scoring scripts)
    dev_prefix: test        # names of the development corpora
    vocab_prefix: vocab      # name of the vocabulary files
    checkpoints: []          # list of checkpoints to load (in this specific order) after main checkpoint
    
    # general
    **encoders:                # this is a list (you can specify several encoders)
      - name: ast             # each encoder or decoder has a name (used for naming variables) and an extension (for files) -> 'code'
        max_len: 500          # max_len of api -> '200'
        attention_type: global
    
    decoders:                # Each encoder or decoder can redefine its own values for a number of parameters,
      - name: nl             # including `cell_size`, `embedding_size` and `attn_size`
        max_len: 30**
    ```
    
- 성능 확인
    
    ```yaml
    # 우리가 테스트 한 것
    step 222000 epoch 50 learning rate 0.306 step-time 1.031 loss 10.060
    test eval: loss 36.92
    starting decoding
    test avg_score=0.1629**(BLUE)**
    
    test 폴더 - avg_score: 0.1626
    ```
    

### 🦑 Hybrid - DeepCom(our model)

- config_hybrid_park.yaml
    
    ```yaml
    # 위 항목 hybrid-deepcom(default)와 동일
    # data
    max_train_size: 0        # maximum size of the training data (0 for unlimited)
    max_dev_size: 0          # maximum size of the dev data
    max_test_size: 0         # maximum size of the test data
    data_dir: ../emse-data(sbt_code_hb)           # directory containing the training data
    model_dir: ../emse-data(sbt_code_hb)/model/hybrid
    train_prefix: train      # name of the training corpus
    script_dir: scripts      # directory where the scripts are kepts (in particular the scoring scripts)
    dev_prefix: test        # names of the development corpora
    vocab_prefix: vocab      # name of the vocabulary files
    checkpoints: []          # list of checkpoints to load (in this specific order) after main checkpoint
    
    # decoding
    score_function: nltk_sentence_bleu # name of the main scoring function, inside 'evaluation.py' (used for selecting models)
    post_process_script: null # path to post-processing script (called before evaluating)
    remove_unk: False        # remove UNK symbols from the decoder output
    beam_size: 5             # beam size for decoding (decoder is greedy by default)
    ensemble: False          # use an ensemble of models while decoding (specified by the --checkpoints parameter)
    output: null             # output file for decoding (writes to standard output by default)
    len_normalization: 1.0   # length normalization coefficient used in beam-search decoder
    early_stopping: True     # reduce beam-size each time a finished hypothesis is encountered (affects decoding speed)
    raw_output: False        # output translation hypotheses without any post-processing
    average: False           # like ensemble, but instead of averaging the log-probs, average all parameters
    pred_edits: False
    
    # general
    **encoders:                # this is a list (you can specify several encoders)
      - name: code             # each encoder or decoder has a name (used for naming variables) and an extension (for files)
        max_len: 200          # max_len of api
        attention_type: global
      - name: ast
        max_len: 500
        attention_type: global
    
    decoders:                # Each encoder or decoder can redefine its own values for a number of parameters,
      - name: nl             # including `cell_size`, `embedding_size` and `attn_size`
        max_len: 30**
    ```
    
- 성능 확인
    
    ```yaml
    step 174000 epoch 50 learning rate 0.306 step-time 1.163 loss 8.117
    test eval: loss 33.40
    starting decoding
    test avg_score=0.3788**(BLEU)**
    
    test 폴더 - avg_score: 0.3798
    ```
    

### 🦑 DeepCom(our model-sim SBT)

- config_park_simSBT.yaml
    
    ```yaml
    # 위 항목 deepcom(default)와 동일
    # data
    max_train_size: 0        # maximum size of the training data (0 for unlimited)
    max_dev_size: 0          # maximum size of the dev data
    max_test_size: 0         # maximum size of the test data
    data_dir: ../emse-data(simsbt_code)           # directory containing the training data
    model_dir: ../emse-data(simsbt_code)/model/default
    train_prefix: train      # name of the training corpus
    script_dir: scripts      # directory where the scripts are kepts (in particular the scoring scripts)
    dev_prefix: test        # names of the development corpora
    vocab_prefix: vocab      # name of the vocabulary files
    checkpoints: []          # list of checkpoints to load (in this specific order) after main checkpoint
    
    # general
    **encoders:                # this is a list (you can specify several encoders)
      - name: ast             # each encoder or decoder has a name (used for naming variables) and an extension (for files) -> 'code'
        max_len: 500          # max_len of api -> '200'
        attention_type: global
    
    decoders:                # Each encoder or decoder can redefine its own values for a number of parameters,
      - name: nl             # including `cell_size`, `embedding_size` and `attn_size`
        max_len: 30**
    ```
    
- 성능 확인
    
    ```yaml
    step 222000 epoch 50 learning rate 0.306 step-time 0.620 loss 9.092
    test eval: loss 36.50
    starting decoding
    test avg_score=0.2424**(BLEU)**
    
    test 폴더 - avg_score: 0.2417
    ```
    

### 🦑 Hybrid - DeepCom(our model-sim SBT)

- config_hybrid_park_simSBT.yaml
    
    ```yaml
    # 위 항목 hybrid-deepcom(default)와 동일
    # data
    max_train_size: 0        # maximum size of the training data (0 for unlimited)
    max_dev_size: 0          # maximum size of the dev data
    max_test_size: 0         # maximum size of the test data
    data_dir: ../emse-data(simsbt_code_hb)           # directory containing the training data
    model_dir: ../emse-data(simsbt_code_hb)/model/hybrid
    train_prefix: train      # name of the training corpus
    script_dir: scripts      # directory where the scripts are kepts (in particular the scoring scripts)
    dev_prefix: test        # names of the development corpora
    vocab_prefix: vocab      # name of the vocabulary files
    checkpoints: []          # list of checkpoints to load (in this specific order) after main checkpoint
    
    # decoding
    score_function: nltk_sentence_bleu # name of the main scoring function, inside 'evaluation.py' (used for selecting models)
    post_process_script: null # path to post-processing script (called before evaluating)
    remove_unk: False        # remove UNK symbols from the decoder output
    beam_size: 5             # beam size for decoding (decoder is greedy by default)
    ensemble: False          # use an ensemble of models while decoding (specified by the --checkpoints parameter)
    output: null             # output file for decoding (writes to standard output by default)
    len_normalization: 1.0   # length normalization coefficient used in beam-search decoder
    early_stopping: True     # reduce beam-size each time a finished hypothesis is encountered (affects decoding speed)
    raw_output: False        # output translation hypotheses without any post-processing
    average: False           # like ensemble, but instead of averaging the log-probs, average all parameters
    pred_edits: False
    
    # general
    **encoders:                # this is a list (you can specify several encoders)
      - name: code             # each encoder or decoder has a name (used for naming variables) and an extension (for files)
        max_len: 200          # max_len of api
        attention_type: global
      - name: ast
        max_len: 500
        attention_type: global
    
    decoders:                # Each encoder or decoder can redefine its own values for a number of parameters,
      - name: nl             # including `cell_size`, `embedding_size` and `attn_size`
        max_len: 30**
    ```
    
- 성능 확인
    
    ```yaml
    step 174000 epoch 50 learning rate 0.306 step-time 0.756 loss 7.984
    test eval: loss 33.54
    starting decoding
    test avg_score=0.3848**(BLEU)**
    
    test 폴더 - avg_score: 0.3849
    ```
    

### 🦑 SeCNN(default)

- Training Details
    
    data_RQ1 데이터 사용하지 않고 오리지널 데이터 사용해 학습
    
- 성능 확인
    
    ```yaml
    After 30000 steps, BLEU in test: 0.32178
    ```
    

### 🦑 SeTransformer(default)

- Training Details
    
    data_RQ1 데이터 사용하지 않고 오리지널 데이터 사용해 학습
    
- 성능 확인
    
    ```yaml
    After 501100 steps
    rate is 0.00001  
    cost is 0.00151 
    In iterator: 229. 
    nowCBleu: 0.41729
    maxCBlue: 0.41747 
    nowSBleu: 0.44359
    maxSBlue: 0.44359
    ```
    

### 🦧 성능 비교 🦧

|  | Deepcom | Deepcom(sbtcode) | H-Deepcom | H-Deepcom(sbtcode) | DeepCom(our model-sim SBTcode) | H-DeepCom(our model-sim SBTcode) | seCNN(default) | seTransformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLEU | 0.2026 | 0.1626 | 0.3820 | 0.3798 | 0.2417 | 0.3849 | 0.32178 | 0.44359 |
| METEOR | 0.3172 | 0.2741 | 0.5126 | 0.5105 | 0.3543 | 0.5164 |  |  |