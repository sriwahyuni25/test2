{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 10,
      "id": "4f372ea0-fc95-403d-bc90-ae8ecdc79b7d",
      "metadata": {
        "id": "4f372ea0-fc95-403d-bc90-ae8ecdc79b7d"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "id": "6a1dfcc6-41fd-4fe3-b5e0-f71f33c6d55c",
      "metadata": {
        "id": "6a1dfcc6-41fd-4fe3-b5e0-f71f33c6d55c"
      },
      "outputs": [],
      "source": [
        "# import data\n",
        "positif = pd.read_csv('pemilu2024_negative.csv')\n",
        "negatif = pd.read_csv('pemilu2024_positive.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "id": "d5c2786f-8b7a-4a1f-b413-c6d01e36d6d9",
      "metadata": {
        "id": "d5c2786f-8b7a-4a1f-b413-c6d01e36d6d9"
      },
      "outputs": [],
      "source": [
        "# merge data\n",
        "data = pd.concat([positif, negatif])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "id": "3830675e-c30b-46b2-acfe-b6f727434e2e",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3830675e-c30b-46b2-acfe-b6f727434e2e",
        "outputId": "52b85d4d-3f1c-47be-bbde-e675035f94fb"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "sentiment\n",
              "negatif    13796\n",
              "positif    10302\n",
              "Name: count, dtype: int64"
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# drop duplicate data\n",
        "data_final = data[['sentiment', 'steming_data']].dropna()\n",
        "data_final.sentiment.value_counts()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "id": "02bac47d-6dbb-421b-9ee2-2ac5465f4206",
      "metadata": {
        "id": "02bac47d-6dbb-421b-9ee2-2ac5465f4206"
      },
      "outputs": [],
      "source": [
        "data_final['sentiment'] = data_final.sentiment.map({'positif': 1, 'negatif': 0})"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "id": "a33ff5fa-aef2-4c54-8520-1a900b9ca2e8",
      "metadata": {
        "id": "a33ff5fa-aef2-4c54-8520-1a900b9ca2e8"
      },
      "outputs": [],
      "source": [
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "id": "26b394d0-375f-4cf6-be9e-d5d515c21d72",
      "metadata": {
        "id": "26b394d0-375f-4cf6-be9e-d5d515c21d72"
      },
      "outputs": [],
      "source": [
        "x = data_final.steming_data\n",
        "y = data_final['sentiment']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "id": "06d55196-e965-43bf-935e-b2c540309a08",
      "metadata": {
        "id": "06d55196-e965-43bf-935e-b2c540309a08"
      },
      "outputs": [],
      "source": [
        "# vectorizing\n",
        "vec = CountVectorizer().fit(x)\n",
        "x_features = vec.get_feature_names_out()\n",
        "x_vec = vec.transform(x)\n",
        "tfidf = TfidfTransformer().fit(x_vec)\n",
        "tfidf_data = tfidf.transform(x_vec)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "id": "c5b24688-4ed6-41ad-8381-52e97ce7da7c",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c5b24688-4ed6-41ad-8381-52e97ce7da7c",
        "outputId": "7b493379-6509-4920-c851-87e5e3be3452"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "  (0, 19697)\t0.3566509379872837\n",
            "  (0, 16970)\t0.36722451381612226\n",
            "  (0, 16932)\t0.2153371788978142\n",
            "  (0, 16699)\t0.17310588617787914\n",
            "  (0, 16146)\t0.3034703297767129\n",
            "  (0, 14780)\t0.3484494354034458\n",
            "  (0, 10097)\t0.519308051100296\n",
            "  (0, 5186)\t0.36722451381612226\n",
            "  (0, 1848)\t0.15653878759060696\n",
            "  (0, 869)\t0.13805860656629193\n",
            "  (1, 21381)\t0.1138030737677894\n",
            "  (1, 20969)\t0.1609765581771452\n",
            "  (1, 19697)\t0.3969120263117456\n",
            "  (1, 19522)\t0.19016355163893583\n",
            "  (1, 19221)\t0.20433960823523695\n",
            "  (1, 17752)\t0.19016355163893583\n",
            "  (1, 16970)\t0.20433960823523695\n",
            "  (1, 16932)\t0.11982292335879642\n",
            "  (1, 16699)\t0.09632360486291805\n",
            "  (1, 16146)\t0.1688640217756324\n",
            "  (1, 14780)\t0.38778468452562026\n",
            "  (1, 14050)\t0.1984560131558728\n",
            "  (1, 14040)\t0.17598749504263472\n",
            "  (1, 10097)\t0.28896546859715694\n",
            "  (1, 8540)\t0.21263206975217394\n",
            "  :\t:\n",
            "  (24094, 230)\t0.16773533803547902\n",
            "  (24095, 21221)\t0.46441955635436566\n",
            "  (24095, 6933)\t0.5251030298581485\n",
            "  (24095, 6783)\t0.32749008169331717\n",
            "  (24095, 6032)\t0.4027736034167126\n",
            "  (24095, 261)\t0.4889835932755085\n",
            "  (24096, 19340)\t0.21929492646078524\n",
            "  (24096, 19315)\t0.17514893962176298\n",
            "  (24096, 18562)\t0.22359493484095455\n",
            "  (24096, 18323)\t0.1482514895479878\n",
            "  (24096, 17860)\t0.6631543670699349\n",
            "  (24096, 15753)\t0.21929492646078524\n",
            "  (24096, 13219)\t0.1334001497089422\n",
            "  (24096, 11359)\t0.1282405804856438\n",
            "  (24096, 10060)\t0.2452054228985081\n",
            "  (24096, 7845)\t0.2452054228985081\n",
            "  (24096, 6729)\t0.41529701321323353\n",
            "  (24096, 6032)\t0.1343457674680164\n",
            "  (24096, 3237)\t0.12799361749359606\n",
            "  (24097, 21236)\t0.3490078272378584\n",
            "  (24097, 14420)\t0.2043090040963889\n",
            "  (24097, 13797)\t0.36665625741980357\n",
            "  (24097, 6032)\t0.3026813541201737\n",
            "  (24097, 3316)\t0.5524484383789728\n",
            "  (24097, 2646)\t0.5524484383789728\n",
            "Shape of Spare Matrix :  (24098, 22056)\n",
            "Amount of Non-Zero occurences :  271740\n"
          ]
        }
      ],
      "source": [
        "print(tfidf_data)\n",
        "print('Shape of Spare Matrix : ', tfidf_data.shape)\n",
        "print('Amount of Non-Zero occurences : ', tfidf_data.nnz)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "id": "a2302dc7-74f8-4477-8495-f3e0a51c3a72",
      "metadata": {
        "id": "a2302dc7-74f8-4477-8495-f3e0a51c3a72"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x_train, x_test, y_train, y_test = train_test_split(tfidf_data, y, random_state=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "id": "a016ad45-b496-47f0-9c30-5a5c20c368b7",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "a016ad45-b496-47f0-9c30-5a5c20c368b7",
        "outputId": "0c7fa99e-2225-40e4-d6ed-e43ffcd12f8c"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<style>#sk-container-id-1 {\n",
              "  /* Definition of color scheme common for light and dark mode */\n",
              "  --sklearn-color-text: black;\n",
              "  --sklearn-color-line: gray;\n",
              "  /* Definition of color scheme for unfitted estimators */\n",
              "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
              "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
              "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
              "  --sklearn-color-unfitted-level-3: chocolate;\n",
              "  /* Definition of color scheme for fitted estimators */\n",
              "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
              "  --sklearn-color-fitted-level-1: #d4ebff;\n",
              "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
              "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
              "\n",
              "  /* Specific color for light theme */\n",
              "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
              "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-icon: #696969;\n",
              "\n",
              "  @media (prefers-color-scheme: dark) {\n",
              "    /* Redefinition of color scheme for dark theme */\n",
              "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
              "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-icon: #878787;\n",
              "  }\n",
              "}\n",
              "\n",
              "#sk-container-id-1 {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 pre {\n",
              "  padding: 0;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-hidden--visually {\n",
              "  border: 0;\n",
              "  clip: rect(1px 1px 1px 1px);\n",
              "  clip: rect(1px, 1px, 1px, 1px);\n",
              "  height: 1px;\n",
              "  margin: -1px;\n",
              "  overflow: hidden;\n",
              "  padding: 0;\n",
              "  position: absolute;\n",
              "  width: 1px;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-dashed-wrapped {\n",
              "  border: 1px dashed var(--sklearn-color-line);\n",
              "  margin: 0 0.4em 0.5em 0.4em;\n",
              "  box-sizing: border-box;\n",
              "  padding-bottom: 0.4em;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-container {\n",
              "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
              "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
              "     so we also need the `!important` here to be able to override the\n",
              "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
              "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
              "  display: inline-block !important;\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-text-repr-fallback {\n",
              "  display: none;\n",
              "}\n",
              "\n",
              "div.sk-parallel-item,\n",
              "div.sk-serial,\n",
              "div.sk-item {\n",
              "  /* draw centered vertical line to link estimators */\n",
              "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
              "  background-size: 2px 100%;\n",
              "  background-repeat: no-repeat;\n",
              "  background-position: center center;\n",
              "}\n",
              "\n",
              "/* Parallel-specific style estimator block */\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item::after {\n",
              "  content: \"\";\n",
              "  width: 100%;\n",
              "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
              "  flex-grow: 1;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel {\n",
              "  display: flex;\n",
              "  align-items: stretch;\n",
              "  justify-content: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
              "  align-self: flex-end;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
              "  align-self: flex-start;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
              "  width: 0;\n",
              "}\n",
              "\n",
              "/* Serial-specific style estimator block */\n",
              "\n",
              "#sk-container-id-1 div.sk-serial {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "  align-items: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  padding-right: 1em;\n",
              "  padding-left: 1em;\n",
              "}\n",
              "\n",
              "\n",
              "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
              "clickable and can be expanded/collapsed.\n",
              "- Pipeline and ColumnTransformer use this feature and define the default style\n",
              "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
              "*/\n",
              "\n",
              "/* Pipeline and ColumnTransformer style (default) */\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable {\n",
              "  /* Default theme specific background. It is overwritten whether we have a\n",
              "  specific estimator or a Pipeline/ColumnTransformer */\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "/* Toggleable label */\n",
              "#sk-container-id-1 label.sk-toggleable__label {\n",
              "  cursor: pointer;\n",
              "  display: block;\n",
              "  width: 100%;\n",
              "  margin-bottom: 0;\n",
              "  padding: 0.5em;\n",
              "  box-sizing: border-box;\n",
              "  text-align: center;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
              "  /* Arrow on the left of the label */\n",
              "  content: \"▸\";\n",
              "  float: left;\n",
              "  margin-right: 0.25em;\n",
              "  color: var(--sklearn-color-icon);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "/* Toggleable content - dropdown */\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content {\n",
              "  max-height: 0;\n",
              "  max-width: 0;\n",
              "  overflow: hidden;\n",
              "  text-align: left;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content pre {\n",
              "  margin: 0.2em;\n",
              "  border-radius: 0.25em;\n",
              "  color: var(--sklearn-color-text);\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
              "  /* Expand drop-down */\n",
              "  max-height: 200px;\n",
              "  max-width: 100%;\n",
              "  overflow: auto;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
              "  content: \"▾\";\n",
              "}\n",
              "\n",
              "/* Pipeline/ColumnTransformer-specific style */\n",
              "\n",
              "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator-specific style */\n",
              "\n",
              "/* Colorize estimator box */\n",
              "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
              "#sk-container-id-1 div.sk-label label {\n",
              "  /* The background is the default theme color */\n",
              "  color: var(--sklearn-color-text-on-default-background);\n",
              "}\n",
              "\n",
              "/* On hover, darken the color of the background */\n",
              "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "/* Label box, darken color on hover, fitted */\n",
              "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator label */\n",
              "\n",
              "#sk-container-id-1 div.sk-label label {\n",
              "  font-family: monospace;\n",
              "  font-weight: bold;\n",
              "  display: inline-block;\n",
              "  line-height: 1.2em;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label-container {\n",
              "  text-align: center;\n",
              "}\n",
              "\n",
              "/* Estimator-specific */\n",
              "#sk-container-id-1 div.sk-estimator {\n",
              "  font-family: monospace;\n",
              "  border: 1px dotted var(--sklearn-color-border-box);\n",
              "  border-radius: 0.25em;\n",
              "  box-sizing: border-box;\n",
              "  margin-bottom: 0.5em;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "/* on hover */\n",
              "#sk-container-id-1 div.sk-estimator:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
              "\n",
              "/* Common style for \"i\" and \"?\" */\n",
              "\n",
              ".sk-estimator-doc-link,\n",
              "a:link.sk-estimator-doc-link,\n",
              "a:visited.sk-estimator-doc-link {\n",
              "  float: right;\n",
              "  font-size: smaller;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1em;\n",
              "  height: 1em;\n",
              "  width: 1em;\n",
              "  text-decoration: none !important;\n",
              "  margin-left: 1ex;\n",
              "  /* unfitted */\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted,\n",
              "a:link.sk-estimator-doc-link.fitted,\n",
              "a:visited.sk-estimator-doc-link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "/* Span, style for the box shown on hovering the info icon */\n",
              ".sk-estimator-doc-link span {\n",
              "  display: none;\n",
              "  z-index: 9999;\n",
              "  position: relative;\n",
              "  font-weight: normal;\n",
              "  right: .2ex;\n",
              "  padding: .5ex;\n",
              "  margin: .5ex;\n",
              "  width: min-content;\n",
              "  min-width: 20ex;\n",
              "  max-width: 50ex;\n",
              "  color: var(--sklearn-color-text);\n",
              "  box-shadow: 2pt 2pt 4pt #999;\n",
              "  /* unfitted */\n",
              "  background: var(--sklearn-color-unfitted-level-0);\n",
              "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted span {\n",
              "  /* fitted */\n",
              "  background: var(--sklearn-color-fitted-level-0);\n",
              "  border: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link:hover span {\n",
              "  display: block;\n",
              "}\n",
              "\n",
              "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link {\n",
              "  float: right;\n",
              "  font-size: 1rem;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1rem;\n",
              "  height: 1rem;\n",
              "  width: 1rem;\n",
              "  text-decoration: none;\n",
              "  /* unfitted */\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "#sk-container-id-1 a.estimator_doc_link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;RandomForestClassifier<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestClassifier.html\">?<span>Documentation for RandomForestClassifier</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>RandomForestClassifier()</pre></div> </div></div></div></div>"
            ],
            "text/plain": [
              "RandomForestClassifier()"
            ]
          },
          "execution_count": 22,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "model = RandomForestClassifier()\n",
        "model.fit(x_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "id": "c25734a9-5d75-4a78-a53a-743d914ab53d",
      "metadata": {
        "id": "c25734a9-5d75-4a78-a53a-743d914ab53d"
      },
      "outputs": [],
      "source": [
        "y_result = model.predict(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "id": "0729b542-f3a2-42d6-8008-2b8967e5c33d",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0729b542-f3a2-42d6-8008-2b8967e5c33d",
        "outputId": "a7e9676c-9eec-499a-e6c6-f2b271a686b7"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[[3073  457]\n",
            " [ 611 1884]]\n",
            " \n",
            "classification Report: \n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.83      0.87      0.85      3530\n",
            "           1       0.80      0.76      0.78      2495\n",
            "\n",
            "    accuracy                           0.82      6025\n",
            "   macro avg       0.82      0.81      0.82      6025\n",
            "weighted avg       0.82      0.82      0.82      6025\n",
            "\n"
          ]
        }
      ],
      "source": [
        "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score\n",
        "print(confusion_matrix(y_test, y_result))\n",
        "print(\" \")\n",
        "\n",
        "print(\"classification Report: \")\n",
        "print(classification_report(y_test, y_result))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "id": "X9_w2pc2ELfQ",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X9_w2pc2ELfQ",
        "outputId": "66cb9267-001d-4f27-b884-d5a35de9c435"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Accuracy Random Forest: 0.82\n"
          ]
        }
      ],
      "source": [
        "accuracy_rf = accuracy_score(y_test, y_result)\n",
        "print(f'Accuracy Random Forest: {accuracy_rf:.2f}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "id": "fba49aad-794f-4257-9e87-48c429fce3fa",
      "metadata": {
        "id": "fba49aad-794f-4257-9e87-48c429fce3fa"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "id": "07ebadf4-e70d-4ee6-b06a-5ea8d9a16610",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "07ebadf4-e70d-4ee6-b06a-5ea8d9a16610",
        "outputId": "5cfa3706-a9e5-4107-b1d5-0a148082cdaa"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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vadixYo6ePDgP459/PhxtW3bVgMHDtTrr7+uEiVKaOPGjerTp4+uXLmSr5tp+L0D9kciCbvz9fVVaGioTVuNGjWUlZWlLVu2qH79+pKks2fPKiEhQeHh4dZ+FSpU0IABAzRgwACNGjVKH3/88XUTSQ8PD2VnZ/9jLPXr11eFChU0d+5cLVu2TP/6179UtGhRSVJ4eLg8PT2VmJioJk2a3MxHBgq9N998U3Xq1FFYWJi17d5779X+/ftz/d6vCQsLU1ZWlnbu3Km6detKuloZ/PNd4Nu3b1dOTo4mTJggN7ery/jnzZtnMw6/d8B5kEjCIe688061a9dO/fr104cffqhixYrphRde0B133KF27dpJkp577jm1atVK1apV0/nz57V27VrVqFHjuuNVqlRJaWlpWr16tWrXri0fH58bVi66dOmi6dOn6+DBg1q7dq21vVixYho2bJiGDBminJwcNWzYUCkpKfrhhx/k7++vqKiogv8igNtUzZo11bVrV02dOtXaNnLkSD344IMaNGiQ+vbtK19fX+3fv19xcXF67733VL16dTVv3lz9+/fXtGnTVLRoUT3//PPy9va2buUVGhqqzMxMvfvuu3r00Uf1ww8/aPr06TbX5vcOOBFHL9JE4fbnBfp/de7cOaN79+5GQECA4e3tbURGRhoHDx60nh80aJBRtWpVw9PT0yhdurTRvXt34/fffzcMI/fNNoZhGAMGDDBKlixpSDJeeeUVwzBsF99fs3//fkOSERISYuTk5Nicy8nJMSZPnmyEhYUZRYsWNUqXLm1ERkYa69evv+nvAridXe+3fOzYMcPDw8P4818lP/74o/HII48Yfn5+hq+vr1GrVi3j9ddft54/efKk0apVK8PT09MICQkx5syZYwQFBRnTp0+39pk4caJRtmxZ678XZs2axe8dcFIWwzAMB+axAAAX9ttvv6lChQpatWqVmjVr5uhwAOQTiSQA4JZZs2aN0tLSVLNmTSUlJWnEiBE6ceKEDh48aF2/COD2wRpJAMAtk5mZqf/85z86evSoihUrpvr16ys2NpYkErhNUZEEAACAKTwiEQAAAKaQSAIAAMAUEkkAAACYQiIJAAAAU0gkAQAAYAqJJICb1rNnT7Vv3976+qGHHtJzzz13y+NYt26dLBaLLly4cMM+FotFCxcuzPOYY8aMUZ06dW4qruPHj8tisWjXrl03NQ4AOBsSSaCQ6tmzpywWiywWizw8PBQaGqpx48YpKyvL7tf+5ptv9Oqrr+apb16SPwCAc2JDcqAQa9mypT7//HNlZGTou+++U3R0tIoWLapRo0bl6nvlyhV5eHgUyHVLlChRIOMAAJwbFUmgEPP09FRwcLBCQkI0cOBANW/eXN9++62k/01Hv/766ypXrpzCwsIkSb/++qs6deqkwMBAlShRQu3atdPx48etY2ZnZ2vo0KEKDAxUyZIlNWLECP31uQZ/ndrOyMjQyJEjVaFCBXl6eio0NFSffvqpjh8/rqZNm0qSihcvLovFop49e0qScnJyFBMTo8qVK8vb21u1a9fW119/bXOd7777TtWqVZO3t7eaNm1qE2dejRw5UtWqVZOPj4+qVKmil19+WZmZmbn6ffjhh6pQoYJ8fHzUqVMnpaSk2Jz/5JNPVKNGDXl5eal69er64IMPbnjN8+fPq2vXripdurS8vb1155136vPPP8937ADgaFQkARfi7e2ts2fPWl+vXr1a/v7+iouLk3T18XWRkZGKiIjQ999/ryJFiui1115Ty5YttWfPHnl4eGjChAmaMWOGPvvsM9WoUUMTJkzQggUL9PDDD9/wuj169FB8fLymTp2q2rVr69ixY/r9999VoUIFzZ8/Xx07dlRCQoL8/f3l7e0tSYqJidEXX3yh6dOn684779SGDRvUrVs3lS5dWk2aNNGvv/6qDh06KDo6Wv3799e2bdv0/PPP5/s7KVasmGbMmKFy5cpp79696tevn4oVK6YRI0ZY+xw+fFjz5s3T4sWLlZqaqj59+ujpp59WbGysJCk2NlajR4/We++9p3vuuUc7d+5Uv3795Ovrq6ioqFzXfPnll7V//34tW7ZMpUqV0uHDh3Xp0qV8xw4ADmcAKJSioqKMdu3aGYZhGDk5OUZcXJzh6elpDBs2zHq+TJkyRkZGhvU9s2fPNsLCwoycnBxrW0ZGhuHt7W2sWLHCMAzDKFu2rDF+/Hjr+czMTKN8+fLWaxmGYTRp0sR49tlnDcMwjISEBEOSERcXd904165da0gyzp8/b227fPmy4ePjY2zatMmmb58+fYynnnrKMAzDGDVqlBEeHm5zfuTIkbnG+itJxoIFC254/u233zbq1q1rff3KK68Y7u7uxm+//WZtW7ZsmeHm5mYkJSUZhmEYVatWNebMmWMzzquvvmpEREQYhmEYx44dMyQZO3fuNAzDMB599FGjV69eN4wBAG4XVCSBQmzJkiXy8/NTZmamcnJy1KVLF40ZM8Z6vmbNmjbrInfv3q3Dhw+rWLFiNuNcvnxZR44cUUpKipKSklSvXj3ruSJFiui+++7LNb19za5du+Tu7q4mTZrkOe7Dhw8rPT1djzzyiE37lStXdM8990iSDhw4YBOHJEVEROT5GtfMnTtXU6dO1ZEjR5SWlqasrCz5+/vb9KlYsaLuuOMOm+vk5OQoISFBxYoV05EjR9SnTx/169fP2icrK0sBAQHXvebAgQPVsWNH7dixQy1atFD79u1Vv379fMcOAI5GIgkUYk2bNtW0adPk4eGhcuXKqUgR25+8r6+vzeu0tDTVrVvXOmX7Z6VLlzYVw7Wp6vxIS0uTJC1dutQmgZOurvssKPHx8eratavGjh2ryMhIBQQE6Msvv9SECRPyHevHH3+cK7F1d3e/7ntatWqlX375Rd99953i4uLUrFkzRUdH65133jH/YQDAAUgkgULM19dXoaGhee5/7733au7cuQoKCspVlbumbNmy2rJlixo3bizpauVt+/btuvfee6/bv2bNmsrJydH69evVvHnzXOevVUSzs7OtbeHh4fL09FRiYuINK5k1atSw3jh0zebNm//5Q/7Jpk2bFBISohdffNHa9ssvv+Tql5iYqJMnT6pcuXLW67i5uSksLExlypRRuXLldPToUXXt2jXP1y5durSioqIUFRWlRo0aafjw4SSSAG473LUNwKpr164qVaqU2rVrp++//17Hjh3TunXrNHjwYP3222+SpGeffVZvvvmmFi5cqJ9//llPP/303+4BWalSJUVFRal3795auHChdcx58+ZJkkJCQmSxWLRkyRKdOXNGaWlpKlasmIYNG6YhQ4Zo5syZOnLkiHbs2KF3331XM2fOlCQNGDBAhw4d0vDhw5WQkKA5c+ZoxowZ+fq8d955pxITE/Xll1/qyJEjmjp1qhYsWJCrn5eXl6KiorR79259//33Gjx4sDp16qTg4GBJ0tixYxUTE6OpU6fq4MGD2rt3rz7//HNNnDjxutcdPXq0Fi1apMOHD2vfvn1asmSJatSoka/YAcAZkEgCsPLx8dGGDRtUsWJFdejQQTVq1FCfPn10+fJla4Xy+eefV/fu3RUVFaWIiAgVK1ZMjz/++N+OO23aND3xxBN6+umnVb16dfXr108XL16UJN1xxx0aO3asXnjhBZUpU0aDBg2SJL366qt6+eWXFRMToxo1aqhly5ZaunSpKleuLOnqusX58+dr4cKFql27tqZPn6433ngjX5/3scce05AhQzRo0CDVqVNHmzZt0ssvv5yrX2hoqDp06KDWrVurRYsWqlWrls32Pn379tUnn3yizz//XDVr1lSTJk00Y8YMa6x/5eHhoVGjRqlWrVpq3Lix3N3d9eWXX+YrdgBwBhbjRivkAQAAgL9BRRIAAACmkEgCAADAFBJJAAAAmEIiCQAAAFNIJAEAAGAKiSQAAABMIZEEAACAKSSSAAAAMIVEEgAAAKaQSAIAAMAUEkkAAACY8n8E9LeCJCiEGQAAAABJRU5ErkJggg==",
            "text/plain": [
              "<Figure size 800x600 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "cm = confusion_matrix(y_test, y_result)\n",
        "\n",
        "# Create a heatmap for the confusion matrix\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
        "            xticklabels=['Positive', 'Negative'],\n",
        "            yticklabels=['Positive', 'Negative'])\n",
        "plt.xlabel('Predicted labels')\n",
        "plt.ylabel('True labels')\n",
        "plt.title('Confusion Matrix')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "id": "bD6yuXe7EXYk",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bD6yuXe7EXYk",
        "outputId": "48cc5f17-4993-469d-a9d8-d1e6026b50e5"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Akurasi Pelatihan Random Forest: 1.00\n",
            "Akurasi Validasi Random Forest: 0.82\n"
          ]
        }
      ],
      "source": [
        "# Menghitung akurasi pelatihan\n",
        "train_accuracy_rf = model.score(x_train, y_train)\n",
        "\n",
        "# Memprediksi label untuk data uji\n",
        "y_pred = model.predict(x_test)\n",
        "\n",
        "validation_accuracy_rf = accuracy_score(y_test, y_pred)\n",
        "\n",
        "print(f'\\nAkurasi Pelatihan Random Forest: {train_accuracy_rf:.2f}')\n",
        "print(f'Akurasi Validasi Random Forest: {validation_accuracy_rf:.2f}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "id": "khjtWGFufKGI",
      "metadata": {
        "id": "khjtWGFufKGI"
      },
      "outputs": [],
      "source": [
        "import pickle\n",
        "\n",
        "# Simpan model ke file\n",
        "with open('random_forest_model.pkl', 'wb') as model_file:\n",
        "    pickle.dump(model, model_file)\n",
        "\n",
        "# Simpan vectorizer\n",
        "with open('vectorizer.pkl', 'wb') as vec_file:\n",
        "    pickle.dump(vec, vec_file)\n",
        "\n",
        "# Simpan tfidf transformer\n",
        "with open('tfidf_transformer.pkl', 'wb') as tfidf_file:\n",
        "    pickle.dump(tfidf, tfidf_file)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "id": "jaloIDNoky8f",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jaloIDNoky8f",
        "outputId": "94a0219e-6c38-4b57-8558-11b780c136fb"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            " * Serving Flask app '__main__'\n",
            " * Debug mode: on\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
            " * Running on http://127.0.0.1:5000\n",
            "Press CTRL+C to quit\n",
            " * Restarting with watchdog (windowsapi)\n"
          ]
        },
        {
          "ename": "SystemExit",
          "evalue": "1",
          "output_type": "error",
          "traceback": [
            "An exception has occurred, use %tb to see the full traceback.\n",
            "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
          ]
        }
      ],
      "source": [
        "from flask import Flask, request, jsonify\n",
        "import pickle\n",
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
        "\n",
        "# Load model\n",
        "with open('random_forest_model.pkl', 'rb') as model_file:\n",
        "    model = pickle.load(model_file)\n",
        "\n",
        "# Load vectorizer and tfidf transformer\n",
        "with open('vectorizer.pkl', 'rb') as vec_file:\n",
        "    vec = pickle.load(vec_file)\n",
        "\n",
        "with open('tfidf_transformer.pkl', 'rb') as tfidf_file:\n",
        "    tfidf = pickle.load(tfidf_file)\n",
        "\n",
        "app = Flask(__name__)\n",
        "\n",
        "@app.route('/', methods=['GET'])\n",
        "def cek():\n",
        "  return jsonify(\"sukses\")\n",
        "@app.route('/predict', methods=['POST'])\n",
        "def predict():\n",
        "    data = request.get_json(force=True)\n",
        "    text = data['text']\n",
        "\n",
        "    # Transform input text\n",
        "    x_vec = vec.transform([text])\n",
        "    tfidf_data = tfidf.transform(x_vec)\n",
        "\n",
        "    # Predict sentiment\n",
        "    prediction = model.predict(tfidf_data)\n",
        "    sentiment = 'positif' if prediction[0] == 1 else 'negatif'\n",
        "\n",
        "    return jsonify({'sentiment': sentiment})\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    app.run(debug=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "-qxWFsh--RTP",
      "metadata": {
        "id": "-qxWFsh--RTP"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
