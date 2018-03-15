---
layout: post
title: Text splitting based on Zipf's Law and Dynamic Programming
date: 2017-07-15 13:32:20 +0300
description: Text splitting(inferring space between mixed words) based on Zipf's Law and Dynamic Programming
img: 2017-07-15-text-splitting/post.jpg # Add image post (optional)
tags: [Blog, NLP]
author: # Add name author (optional)
---

> "OCRjustmessedupmysentence!Wheredidthespacesgo?Ireallyneedsomespaces!"
> 
> ???  ---  "What are you talking about?"

What you see above is a sentence without spaces. This case is common when dealing with the Optical Character Recognition(OCR) result due to the poor scan quality and weird fonts.

This tiny script is used for adding the correct spaces between words that are mistakenly connected. For example, when the space in the sentence "thomson reuters" is lost, the sentence becomes "thomsonreuters". In order to utilize the text content, we want to recover the correct spaces between words.

> Wait… why not just match the word as long as possible if it is in the dictionary ?

Yes, it is a possible solution. However, the strategy is hard to design, the word "on edge" can be "one dge" if we just follows the longest match. Further more, the general rules for the cases with "a", "the", "at", "on" are very hard to design. Therefore, we need the help of math. Surprisingly, this problem can be solved by applying a simple rule.

To describe the idea in one sentence: **by adding the space between words to maximize the probability(or minimize the cost) of occurrence of the word series**.

First, in order to define the cost of a word, we introduce the `Zipf's Law`:

> From [Wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law):
>
> Zipf's law states that given a large sample of words used, the frequency of any word is **inversely** proportional to its rank in the frequency table. 

According to Zipf's Law, we can make these assumptions: 

$$Prob(w) \approx \frac{1}{R_w log N}, Cost(w) = \frac{1}{Prob(w)} \approx R_w log N$$, where $$w$$ is the word, $$R_w$$ is its rank in the frequency table, $$N$$ is the size of the word dictionary. 

* if the word does not exist in the table, we can set the cost to infinity or pre-set value(to encourage new words).

Once we set up the cost for arbitrary word, the rest job is adding spaces to minimize the cost. For example, the cost of "onetwo" is infinity while the cost of "one two" is the minima. Among all the possible candidates, there should be one that can perfectly separate all words while the cost is minimized. 

> Hmmm, Let me guess, dynamic programming ?

Right, the best solution is dynamic programming. It is quite obvious that we do not need to try every possible splitting, at each possible space position, we can based on previous best split. Here, to keep it simple, we assume the maximum length of word is `3`, and let the previous words are separated by our rules. Therefore, we only need to look back 3 characters to find best split. This figure is the illustration for case "...onetwo": (the text before is omitted)

<img src="/assets/img/2017-07-15-text-splitting/text_split.jpg" alt="A toy example for demostration" width="50%" style="margin-left:25%;">

if we print the cost list for "onetwo", we get:

```
[('o', 8.703316786964155)]
[('n', 7.810148472510853), ('on', 5.2356296640331665)]
[('e', 7.8746869936484245), ('ne', 9.651453678290338), ('one', 6.018389003282799)]
[('t', 7.819627216465397), ('et', 10.18483271861092), ('net', 10.534571835350045)]
[('w', 9.228079918573927), ('tw', 12.71746701641631), ('etw', inf)]
[('o', 8.703316786964155), ('wo', 12.729573108775922), ('two', 6.523483952339804)]
```
Among these logs, we can find that the best solution is "one two", where

$$minima = Cost['one']+Cost['two'] \approx 6.018+6.523=12.541$$

As for the complexity analysis, let $$N$$ be the length of the input string, $$M$$ be the maximum length of word we need to consider. This algorithm just runs over each slot between characters and look back $$M$$ characters to generate possible words, it calculates the min cost and records the split position and goes to the next. The running time of this algorithm is linear as $$O(NM)$$.

After we finish the process, we can backtrace the path and output the sentence.

```
Cost: [0, 8.703316786964155, 5.2356296640331665, 6.018389003282799, 13.838016219748196, 18.735856019699106, 12.541872955622603]
Cut_point: [1, 2, 3, 1, 2, 3]
```
We begin with the last element of `Cut_point` list, the value indicates the length of the word. Then we jump 3 characters to `e` and add a space behind it. Meanwhile, the value at `e` is `Cut_point[2] = 3`, we follow it and come the beginning and the output sentence is "one two".

> Talk is cheap, show me the code

In terms of the real implement, there are something to figure out.

* The maximum length of word: this constant is important as for the speed. Here we set the number as `16`.
* Handling the punctuations, here we just ignore them and remain them same in the output.
* We need a word frequency dictionary. You can find the file [here](http://tinypaste.com/c1666a6b) (it is build on a small subset of Wikipedia).
* In each iteration, once we find the best split, we record the min cost as well as the position of space for outputting the sentence.

{% gist 8eec0630fbaba23954b982a664051b37 %}

Here is a test case for the code.

```python
# test case
cleaner = TextCleaner(dict_path='./word-frequency.txt', word_max_length=16)
origin_str = "They're all converging today at a Rhode Island convention centre where about three-dozen U.S. state governors are holding their annual summer meeting, with trade uncertainty looming ahead."
no_space_str = re.sub(r'\s+', '', origin_str)
>>>
"They'reallconvergingtodayataRhodeIslandconventioncentrewhereaboutthree-dozenU.S.stategovernorsareholdingtheirannualsummermeeting,withtradeuncertaintyloomingahead."
cleaner.split_sentences(no_space_str)
>>>
"They're all converging today at a Rhode Island convention centre where about three-dozen U.S.state governors are holding their annual summer meeting,with trade uncertainty looming ahead."
```

Back to the beginning, if we take the weird sentence as the input:

```python
cleaner.split_sentences("OCRjustmessedupmysentence!Wheredidthespacesgo?Ireallyneedsomespaces!")
>>>
'OCR just messed up my sentence!Where did the spaces go?I really need some spaces!'
```

What's more, as kind of an extension, it supports user's dictionary. Just add the new word to the beginning part of `word-frequency.txt`, according to this algorithm, the cost of the new word should be the minima which results in the correct extraction.

> Well then, what is the weakness?

**--- There is no free lunch.**

Although this script is easy with relative good performance, we should realize that it is also limited:

* In the OCR results, it is very common that the word could be mis-recognized(l and I, 0 and O, etc.). As this algorithm is based on the correct spelling of the word, it under-performs if there are typos in spelling.
* Currently, this method just split the sentence by the punctuations, it cannot fit in some special case like email address and telephone number.
* The quality of splitting depends on the `word-frequency.txt` (corpus).
* Currently, we can not fix the word that is separated by line change.

**Note**: this post is my personal understanding based on this answer on [StackOverflow](https://stackoverflow.com/questions/2058925/how-can-i-break-up-this-long-line-in-python).