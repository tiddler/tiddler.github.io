---
layout: post
title: Text splitting(inferring space between mixed words) based on Zipf's Law and Dynamic Programming
---

## Text splitting(inferring space between mixed words) based on Zipf's Law and Dynamic Programming

> OCRjustmessesupmysentence!Wheredidthespacesgo?Ireallyneedsomespaces!

What you see above is a sentence without spaces. This case is common when dealing with the Optical Character Recognition(OCR) result due to the poor scan quality and wired fonts.

This tiny script is used for adding the correct spaces between words that are mistakenly connected. For example, when the space in the sentence "thomson reuters" is lost, the sentence becomes "thomsonreuters".  In order to utilize the text content, we want to recover the correct spaces between words.

> Wait… why not just match the word as long as possible if it is in the dictionary ?

Yes, it is a possible solution. However, the strategy is hard to design, the word "on edge" can be "one dge" if we just follows the longest match. Further more, the general rules for the cases with "a", "the", "at", "on" are very hard to design. Therefore, we need the help of math.

To describe the idea in one sentence: **by adding the space between words to maximize the probability(or minimize the cost) of occurrence of the word series**.

First, in order to define the cost of a word, we introduce the `Zipf's Law`

> From [Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law):
>
> Zipf's law states that given a large sample of words used, the frequency of any word is **inversely** proportional to its rank in the frequency table. 

According to Zipf's Law, we can make these assumptions: 

$Prob(w) \approx \frac{1}{R_w log N}, Cost(w) = \frac{1}{Prob(w)} \approx R_w log N$, where $w$ is the word, $R_w$ is its rank in the frequency table, N is the size of the word dictionary. 

*if the word does not exist in the table, we can set the cost to infinity or pre-set value(to encourage new words).

Once we set up the cost for arbitrary word, the rest job is adding spaces to minimize the cost. For example, the cost of "onetwo" is infinity while the cost of "one two" is the minima. Among all the possible adding solutions, there should be one that can perfectly separate all words while the cost should be minimized. 

> Hmmm, Let me guess, dynamic programming ?

Right, the best solution is dynamic programming. It is quite obvious that we do not need to try every possible splitting, at each possible space position, we can based on previous best split. Here, we assume the maximum length of word is `3`, therefore, we only look back 3 characters to find best split. Here is the illustration for case "...onetwo"(the text before is omitted)

![A toy example for demostration](/img/post_1_text_split.jpg)

if we print the cost list for "onetwo", we get

```
[('o', 8.703316786964155)]
[('n', 7.810148472510853), ('on', 5.2356296640331665)]
[('e', 7.8746869936484245), ('ne', 9.651453678290338), ('one', 6.018389003282799)]
[('t', 7.819627216465397), ('et', 10.18483271861092), ('net', 10.534571835350045), ('onet', inf)]
[('w', 9.228079918573927), ('tw', 12.71746701641631), ('etw', inf), ('netw', inf), ('onetw', inf)]
[('o', 8.703316786964155), ('wo', 12.729573108775922), ('two', 6.523483952339804), ('etwo', inf), ('netwo', inf), ('onetwo', inf)]
```

The running time of this algorithm is linear.

> Talk is cheap, show me the code

In terms of the real implement, there are something to figure out.

* The maximum length of word: this constant is important as for the speed. Here we set the number as `16`.
* Handling the punctuations, here we just ignore them and remain them same in the output.
* We need a word frequency dictionary. You can find the file [here](http://tinypaste.com/c1666a6b) (it is build on a small subset of Wikipedia).
* In each iteration, once we find the best split, we record the min cost as well as the position of space for outputting the sentence.

```Python
class TextCleaner:
    def __init__(self, dict_path='./word-frequency.txt', word_max_length=16):
        """
        initialize the class given a path to dictionary and maximal length of single word
        :param dict_path('./word-frequency.txt'): path to dictionary file 
        :param word_max_length(16): the length of longest word in the document
        """
        self.word_max_length = word_max_length
        try:
            with open(dict_path, 'r') as dict_file:
                self.words = dict_file.read().split()
                word_num = len(self.words)
                self.word_cost = dict(
                    (word, log((idx + 1) * log(word_num)))
                    for idx, word in enumerate(self.words)
                )
        except IOError as e:
            print('Error in loading file: ', e)
            raise e
        except:
            raise
        else:
            print('Successfully load the word list')

    def __infer_space(self, cost_list, sentence, i):
        """
        calculate all possible costs by adding a space in [i - word_max_length: i]
        and return the best split with minimal cost value
        :param cost_list: list of cost at each slice point
        :param sentence: the sentence to add space
        :param i: the index of position to add space
        :return: Tuple: minimal cost with responding slice position 
        """
        candidates = enumerate(reversed(cost_list[max(0, i - self.word_max_length): i]))
        return min(
            # return the lowest cost with the slice position
            # if the word does not exist in the dictionary, set the cost to inf
            (cost + self.word_cost.get(sentence[i - length - 1: i].lower(), float('inf')), length + 1)
            for length, cost in candidates
        )

    def split_sentence(self, sentence):
        """
        split the sentence by adding  possible spaces
        :param sentence: String
        :return: String
        """
        sentence = re.sub('\W', '', sentence)
        word_list = []
        cost_list = [0]
        cut_point = []

        for i in range(1, len(sentence) + 1):
            cost, point = self.__infer_space(cost_list, sentence, i)
            cost_list.append(cost)
            cut_point.append(point)

        i = len(cut_point)
        while i > 0:
            point = cut_point[i - 1]
            word_list.append(sentence[i - point: i])
            i -= point
        return " ".join(reversed(word_list))

    def split_paragraph(self, paragraph):
        """
        split the paragraph by non-word characters and call split_string() to add spaces
        :param paragraph: String
        :return: String
        """

        # customized mapper function that remain punctuations same and split sentences
        def mapper(sen):
            # if the "sen" is spaces or punctuations, remain the same
            if len(sen) == 0 or re.search(r'\W|\d', sen) is not None:
                return sen
            else:
                return self.split_sentence(sen)

        # remove all Unicode characters and replace multi-spaces with single one.
        paragraph = re.sub(r'\s+', ' ', paragraph.encode('ascii', 'ignore'))
        # use re.split so that we can keep the delimiters
        sentence_list = re.split('(\W+|\d+)', paragraph)
        return "".join(map(mapper, sentence_list))
```

```python
# test case
cleaner = TextCleaner(dict_path='./word-frequency.txt', word_max_length=16)
origin_str = "They're all converging today at a Rhode Island convention centre where about three-dozen U.S. state governors are holding their annual summer meeting, with trade uncertainty looming ahead."
no_space_str = re.sub(r'\s+', '', origin_str)
>>>"They'reallconvergingtodayataRhodeIslandconventioncentrewhereaboutthree-dozenU.S.stategovernorsareholdingtheirannualsummermeeting,withtradeuncertaintyloomingahead."
cleaner.split_sentences(no_space_str)
>>>
"They're all converging today at a Rhode Island convention centre where about three-dozen U.S.state governors are holding their annual summer meeting,with trade uncertainty looming ahead."
```

Back to the beginning, if we take the sentence as the input

```python
cleaner.split_sentences("OCRjustmessesupmysentence!Wheredidthespacesgo?Ireallyneedsomespaces!")
>>>
'OCR just messes up my sentence!Where did the spaces go?I really need some spaces!'
```

What's more, as kind of an extension, it supports user's dictionary. Just add the new word to the beginning part of `word-frequency.txt`, according to this algorithm, the cost of the new word should be the minima.

> Well then, what is the weakness?

**There is no free lunch. **

Although this script is easy with relative good performance, we should realize that it is also limited:

* In the OCR results, it is very common that the word could be mis-recognized(l and I, 0 and O, etc.). As this algorithm is based on the correct spelling of the word, it under-performs if there are typos in spelling.
* Currently, this method just split the sentence by the punctuations, it cannot fit in some special case like email address and telephone number.
* The quality of splitting depends on the `word-frequency.txt` (corpus).
* Currently, we can not fix the word that is separated by line change.