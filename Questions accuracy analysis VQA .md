# Accuracy Comparison (VQA with KB vs Without KB)
    ############################################# OVERALL #############################################
    ╒═══════════════╤══════════════════╤════════════════════════════╕
    │   overall[KB] │   overall[NO KB] │   difference with baseline │
    ╞═══════════════╪══════════════════╪════════════════════════════╡
    │        61.549 │           61.216 │                      0.332 │
    ╘═══════════════╧══════════════════╧════════════════════════════╛
    
    
    ############################################# PERANSWERTYPE #############################################
    ╒═══════════╤═════════════════════╤════════════════════════╤════════════════════════════╕
    │ Type      │   perAnswerType[KB] │   perAnswerType[NO KB] │   difference with baseline │
    ╞═══════════╪═════════════════════╪════════════════════════╪════════════════════════════╡
    │ number ++ │              42.554 │                 40.782 │                      1.773 │
    ├───────────┼─────────────────────┼────────────────────────┼────────────────────────────┤
    │ other ++  │              57.293 │                 56.101 │                      1.192 │
    ├───────────┼─────────────────────┼────────────────────────┼────────────────────────────┤
    │ yes/no -- │              73.767 │                 75.066 │                     -1.299 │
    ╘═══════════╧═════════════════════╧════════════════════════╧════════════════════════════╛
    
    
    ############################################# PERQUESTIONTYPE #############################################
    ╒═════════════════════════════╤═══════════════════════╤══════════════════════════╤════════════════════════════╕
    │ Type                        │   perQuestionType[KB] │   perQuestionType[NO KB] │   difference with baseline │
    ╞═════════════════════════════╪═══════════════════════╪══════════════════════════╪════════════════════════════╡
    │ are --                      │                69.453 │                   70.945 │                     -1.492 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ are the --                  │                72.635 │                   73.741 │                     -1.106 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ are there --                │                74.753 │                   76.185 │                     -1.433 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ are there any +             │                77.286 │                   76.82  │                      0.466 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ are these --                │                73.727 │                   75.294 │                     -1.567 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ are they ----               │                70.742 │                   74.846 │                     -4.105 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ can you ---                 │                67.695 │                   70.103 │                     -2.408 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ could -                     │                81.359 │                   81.796 │                     -0.437 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ do +                        │                73.38  │                   73.154 │                      0.226 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ do you ++                   │                74.503 │                   73.246 │                      1.257 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ does the ++                 │                73.629 │                   72.611 │                      1.018 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ does this -                 │                73.956 │                   74.297 │                     -0.341 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ has --                      │                70.169 │                   71.364 │                     -1.195 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ how -                       │                31.573 │                   31.808 │                     -0.235 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ how many ++                 │                47.075 │                   45.581 │                      1.494 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ how many people are +++     │                47.282 │                   45.172 │                      2.11  │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ how many people are in -    │                46.829 │                   47.635 │                     -0.807 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is -                        │                71.237 │                   71.695 │                     -0.458 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is he ----                  │                72.042 │                   75.078 │                     -3.036 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is it +                     │                81.649 │                   81.352 │                      0.297 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is that a ---               │                69.356 │                   71.485 │                     -2.129 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is the -                    │                73.278 │                   74.189 │                     -0.911 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is the man ---              │                70.016 │                   72.987 │                     -2.971 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is the person ----          │                70.907 │                   74.244 │                     -3.338 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is the woman -              │                72.056 │                   72.681 │                     -0.625 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is there +                  │                74.163 │                   74.144 │                      0.019 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is there a -                │                73.013 │                   73.325 │                     -0.312 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is this --                  │                73.835 │                   75.048 │                     -1.213 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is this a ---               │                73.029 │                   75.638 │                     -2.609 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is this an --               │                72.562 │                   73.73  │                     -1.169 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ is this person ---          │                70.163 │                   73.011 │                     -2.847 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ none of the above +         │                57.913 │                   57.705 │                      0.208 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ was --                      │                77.078 │                   79.059 │                     -1.98  │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what ++                     │                50.309 │                   48.95  │                      1.359 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what animal is +++          │                79.772 │                   76.783 │                      2.989 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what are +                  │                59.434 │                   59.299 │                      0.135 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what are the +++            │                53.525 │                   51.1   │                      2.425 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what brand +++              │                51.979 │                   49.465 │                      2.513 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what color ++               │                67.556 │                   65.938 │                      1.618 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what color are the +        │                71.058 │                   70.523 │                      0.536 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what color is +             │                76.861 │                   76.165 │                      0.697 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what color is the +         │                75.603 │                   74.796 │                      0.807 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what does the ++            │                29.462 │                   28.264 │                      1.198 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is ++                  │                48.668 │                   47.318 │                      1.35  │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is in the +            │                55.222 │                   54.564 │                      0.658 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is on the +            │                47.847 │                   47.01  │                      0.837 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is the ++              │                53.189 │                   51.799 │                      1.391 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is the color of the ++ │                79.322 │                   78.123 │                      1.199 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is the man +++         │                62.475 │                   59.786 │                      2.689 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is the name +++        │                20.218 │                   17.333 │                      2.885 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is the person +++      │                65.589 │                   62.722 │                      2.867 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is the woman ++        │                55.791 │                   53.998 │                      1.794 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what is this ++             │                65.719 │                   64.505 │                      1.215 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what kind of +              │                59.178 │                   59.151 │                      0.027 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what number is +            │                19.376 │                   18.93  │                      0.446 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what room is +              │                92.467 │                   92.139 │                      0.328 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what sport is +             │                90.175 │                   89.355 │                      0.82  │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what time ++++              │                34.96  │                   28.929 │                      6.031 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ what type of +              │                59.866 │                   58.95  │                      0.916 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ where are the ++            │                39.741 │                   37.822 │                      1.919 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ where is the ++             │                35.832 │                   34.623 │                      1.208 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ which +                     │                49.91  │                   49.138 │                      0.772 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ who is +++                  │                42.692 │                   39.729 │                      2.963 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ why +                       │                23.081 │                   22.955 │                      0.125 │
    ├─────────────────────────────┼───────────────────────┼──────────────────────────┼────────────────────────────┤
    │ why is the +                │                23.988 │                   23.852 │                      0.136 │
    ╘═════════════════════════════╧═══════════════════════╧══════════════════════════╧════════════════════════════╛
    
    
    

    ############################################# Better Categories [41] #############################################
    ╒═════════════════════════════╤═══════════╕
    │ Question Type               │      gain │
    ╞═════════════════════════════╪═══════════╡
    │ are there any +             │ 0.466165  │
    ├─────────────────────────────┼───────────┤
    │ do +                        │ 0.226214  │
    ├─────────────────────────────┼───────────┤
    │ do you ++                   │ 1.25691   │
    ├─────────────────────────────┼───────────┤
    │ does the ++                 │ 1.01791   │
    ├─────────────────────────────┼───────────┤
    │ how many ++                 │ 1.4935    │
    ├─────────────────────────────┼───────────┤
    │ how many people are +++     │ 2.10973   │
    ├─────────────────────────────┼───────────┤
    │ is it +                     │ 0.297252  │
    ├─────────────────────────────┼───────────┤
    │ is there +                  │ 0.0192308 │
    ├─────────────────────────────┼───────────┤
    │ none of the above +         │ 0.208187  │
    ├─────────────────────────────┼───────────┤
    │ what ++                     │ 1.35938   │
    ├─────────────────────────────┼───────────┤
    │ what animal is +++          │ 2.9892    │
    ├─────────────────────────────┼───────────┤
    │ what are +                  │ 0.134961  │
    ├─────────────────────────────┼───────────┤
    │ what are the +++            │ 2.42535   │
    ├─────────────────────────────┼───────────┤
    │ what brand +++              │ 2.51337   │
    ├─────────────────────────────┼───────────┤
    │ what color ++               │ 1.61765   │
    ├─────────────────────────────┼───────────┤
    │ what color are the +        │ 0.5356    │
    ├─────────────────────────────┼───────────┤
    │ what color is +             │ 0.696629  │
    ├─────────────────────────────┼───────────┤
    │ what color is the +         │ 0.807197  │
    ├─────────────────────────────┼───────────┤
    │ what does the ++            │ 1.19797   │
    ├─────────────────────────────┼───────────┤
    │ what is ++                  │ 1.34956   │
    ├─────────────────────────────┼───────────┤
    │ what is in the +            │ 0.657819  │
    ├─────────────────────────────┼───────────┤
    │ what is on the +            │ 0.837167  │
    ├─────────────────────────────┼───────────┤
    │ what is the ++              │ 1.39082   │
    ├─────────────────────────────┼───────────┤
    │ what is the color of the ++ │ 1.19855   │
    ├─────────────────────────────┼───────────┤
    │ what is the man +++         │ 2.6887    │
    ├─────────────────────────────┼───────────┤
    │ what is the name +++        │ 2.88462   │
    ├─────────────────────────────┼───────────┤
    │ what is the person +++      │ 2.86667   │
    ├─────────────────────────────┼───────────┤
    │ what is the woman ++        │ 1.79367   │
    ├─────────────────────────────┼───────────┤
    │ what is this ++             │ 1.21462   │
    ├─────────────────────────────┼───────────┤
    │ what kind of +              │ 0.0273973 │
    ├─────────────────────────────┼───────────┤
    │ what number is +            │ 0.445765  │
    ├─────────────────────────────┼───────────┤
    │ what room is +              │ 0.328084  │
    ├─────────────────────────────┼───────────┤
    │ what sport is +             │ 0.819521  │
    ├─────────────────────────────┼───────────┤
    │ what time ++++              │ 6.03093   │
    ├─────────────────────────────┼───────────┤
    │ what type of +              │ 0.915842  │
    ├─────────────────────────────┼───────────┤
    │ where are the ++            │ 1.91927   │
    ├─────────────────────────────┼───────────┤
    │ where is the ++             │ 1.20829   │
    ├─────────────────────────────┼───────────┤
    │ which +                     │ 0.772059  │
    ├─────────────────────────────┼───────────┤
    │ who is +++                  │ 2.96262   │
    ├─────────────────────────────┼───────────┤
    │ why +                       │ 0.125174  │
    ├─────────────────────────────┼───────────┤
    │ why is the +                │ 0.136187  │
    ╘═════════════════════════════╧═══════════╛
    
    
    ############################################# Worst Categories [24] #############################################
    ╒══════════════════════════╤═══════════╕
    │ Question Type            │      gain │
    ╞══════════════════════════╪═══════════╡
    │ are --                   │ -1.49216  │
    ├──────────────────────────┼───────────┤
    │ are the --               │ -1.10562  │
    ├──────────────────────────┼───────────┤
    │ are there --             │ -1.4327   │
    ├──────────────────────────┼───────────┤
    │ are these --             │ -1.56745  │
    ├──────────────────────────┼───────────┤
    │ are they ----            │ -4.10487  │
    ├──────────────────────────┼───────────┤
    │ can you ---              │ -2.40826  │
    ├──────────────────────────┼───────────┤
    │ could -                  │ -0.436893 │
    ├──────────────────────────┼───────────┤
    │ does this -              │ -0.341266 │
    ├──────────────────────────┼───────────┤
    │ has --                   │ -1.1945   │
    ├──────────────────────────┼───────────┤
    │ how -                    │ -0.235343 │
    ├──────────────────────────┼───────────┤
    │ how many people are in - │ -0.80663  │
    ├──────────────────────────┼───────────┤
    │ is -                     │ -0.457558 │
    ├──────────────────────────┼───────────┤
    │ is he ----               │ -3.03588  │
    ├──────────────────────────┼───────────┤
    │ is that a ---            │ -2.12885  │
    ├──────────────────────────┼───────────┤
    │ is the -                 │ -0.911092 │
    ├──────────────────────────┼───────────┤
    │ is the man ---           │ -2.97093  │
    ├──────────────────────────┼───────────┤
    │ is the person ----       │ -3.33753  │
    ├──────────────────────────┼───────────┤
    │ is the woman -           │ -0.625    │
    ├──────────────────────────┼───────────┤
    │ is there a -             │ -0.312032 │
    ├──────────────────────────┼───────────┤
    │ is this --               │ -1.21286  │
    ├──────────────────────────┼───────────┤
    │ is this a ---            │ -2.60945  │
    ├──────────────────────────┼───────────┤
    │ is this an --            │ -1.16854  │
    ├──────────────────────────┼───────────┤
    │ is this person ---       │ -2.84741  │
    ├──────────────────────────┼───────────┤
    │ was --                   │ -1.98044  │
    ╘══════════════════════════╧═══════════╛
    
    
    


