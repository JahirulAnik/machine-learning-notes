# Machine Learning Notes

Machine learning algorithms explained **step by step**: free, open notes written in [Obsidian](https://obsidian.md).

Every lesson answers three questions in the same order:

1. **Why do we need this algorithm?**
2. **What does it look like in real life?** (one worked example used through the whole note)
3. **How does it work, one step at a time?** (formulas, hand calculations, then code)

## Lessons

| # | Note | Type | Idea in one line |
|---|---|---|---|
| 1 | [Linear Regression](Linear%20Regression.md) | Regression | Fit a straight line, measure the error (MSE), and use derivatives and gradient descent to improve it. |
| 2 | [Logistic Regression](Logistic%20Regression.md) | Classification | Turn a linear score into a probability with the sigmoid, and fit it by maximising the likelihood. |
| 3 | [K-Nearest Neighbors](K-Nearest%20Neighbors.md) | Classification and regression | Measure the distance to every stored example, take the k nearest, and let them vote or average. |

More algorithms will be added over time.

## How each note is organised

1. Why we need it
2. Real-life example
3. Explanation step by step
4. Many features (matrix / vector form)
5. How good is the model? (evaluation)
6. Assumptions, limits and fixes
7. Code (from scratch and with scikit-learn)
8. Common mistakes
9. Summary and cheat sheet
10. Exercises

## How to read these notes

- **Best experience: Obsidian.** Clone this repository, then in Obsidian choose *Open folder as vault* and select it. Links between notes, callouts, images and math all render as intended.
- **On GitHub:** the notes are readable, but Obsidian-specific features (`[[links]]`, `![[images]]`, callouts) will not display properly.

```bash
git clone https://github.com/JahirulAnik/machine-learning-notes.git
```

To follow along, school-level algebra and a little Python help. The code examples use NumPy and scikit-learn.

## Feedback and corrections

Found a mistake or something unclear? Please [open an issue](https://github.com/JahirulAnik/machine-learning-notes/issues). Corrections are very welcome.

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt it, including commercially, as long as you give credit. See [LICENSE](LICENSE).

Created by [JahirulAnik](https://github.com/JahirulAnik).
