from manim import (
    Text,
    VGroup,
    SurroundingRectangle,
    Arrow,
    Create,
    Write,
    FadeOut,
    Transform,
    Scene,
    np,
    RIGHT,
    UP,
    DOWN,
    LEFT,
    WHITE,
    GREEN,
    YELLOW,
    RED,
)
import numpy as np


class ChunkingAnalysisAnimation(Scene):
    def construct(self):
        # Scene 1: Introduction
        title = Text(
            "Optimizing Chunks in Chain-of-Density Summarization", font_size=40
        )
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Scene 2: Visualizing Scores Over Iterations
        score_group, score_boxes = self.show_score_progression()

        # Scene 3: Analyzing Optimal Improvement Range
        self.show_optimal_improvement_range(score_group)

        # Scene 4: Finding Optimal Score Range
        self.show_optimal_score_range(score_group, score_boxes)

        # Scene 5: Conclusion
        self.show_conclusion()

    def show_score_progression(self):
        scores = [3.0, 2.5, 4.0, 3.8, 5.5, 4.2, 6.8, 5.1, 7.0, 6.3, 8.3, 7.5]
        score_texts = [Text(f"{score:.1f}", font_size=24) for score in scores]
        score_group = VGroup(*score_texts).arrange(RIGHT, buff=0.7).to_edge(UP, buff=1)

        # Create boxes around each score
        score_boxes = VGroup(
            *[SurroundingRectangle(text, color=WHITE) for text in score_texts]
        )

        self.play(Write(score_group), Create(score_boxes))
        self.wait(2)

        explanation = Text("Scores over chunks", font_size=24)
        explanation.next_to(score_group, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))

        return score_group, score_boxes

    def show_optimal_improvement_range(self, score_group):
        scores = [3.0, 2.5, 4.0, 3.8, 5.5, 4.2, 6.8, 5.1, 7.0, 6.3, 8.3, 7.5]
        improvements = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        improvement_texts = [Text(f"{imp:+.1f}", font_size=24) for imp in improvements]
        improvement_group = (
            VGroup(*improvement_texts)
            .arrange(RIGHT, buff=0.7)
            .next_to(score_group, DOWN, buff=1.5)
        )

        # Create boxes around each improvement
        improvement_boxes = VGroup(
            *[SurroundingRectangle(text, color=WHITE) for text in improvement_texts]
        )

        self.play(Write(improvement_group), Create(improvement_boxes))
        self.wait(2)

        # Calculate moving average of improvements
        window_size = 3
        moving_avg = [
            np.mean(improvements[i : i + window_size])
            for i in range(len(improvements) - window_size + 1)
        ]
        threshold = 0.5 * np.mean(improvements)  # 50% of mean improvement

        # Find the optimal range
        optimal_start = next(
            (i for i, avg in enumerate(moving_avg) if avg >= threshold), 0
        )
        optimal_end = next(
            (
                i
                for i in range(len(moving_avg) - 1, -1, -1)
                if moving_avg[i] >= threshold
            ),
            len(moving_avg) - 1,
        )

        optimal_range_box = SurroundingRectangle(
            improvement_group[optimal_start : optimal_end + 1], color=GREEN
        )
        self.play(Create(optimal_range_box))

        arrow = Arrow(
            start=optimal_range_box.get_bottom(),
            end=optimal_range_box.get_bottom() + DOWN,
            color=GREEN,
        )
        self.play(Create(arrow))

        explanation = Text(
            f"Optimal improvement range: chunks {optimal_start}-{optimal_end+2}",
            font_size=24,
        )
        explanation.next_to(arrow, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)
        self.play(
            FadeOut(improvement_group),
            FadeOut(improvement_boxes),
            FadeOut(optimal_range_box),
            FadeOut(arrow),
            FadeOut(explanation),
        )

    def show_optimal_score_range(self, score_group, score_boxes):
        scores = [3.0, 2.5, 4.0, 3.8, 5.5, 4.2, 6.8, 5.1, 7.0, 6.3, 8.3, 7.5]
        highest_score = max(scores)
        highest_score_index = scores.index(highest_score)

        # Highlight the highest score
        highest_score_box = score_boxes[highest_score_index].copy().set_color(YELLOW)
        self.play(
            score_group[highest_score_index].animate.set_color(YELLOW),
            Transform(score_boxes[highest_score_index], highest_score_box),
        )
        self.wait(1)

        # Find the range with the highest cumulative improvement leading to the highest score
        best_start = 0
        best_end = highest_score_index
        best_improvement = sum(
            scores[i + 1] - scores[i] for i in range(highest_score_index)
        )
        for start in range(highest_score_index):
            current_improvement = sum(
                scores[i + 1] - scores[i] for i in range(start, highest_score_index)
            )
            if current_improvement > best_improvement:
                best_start = start
                best_improvement = current_improvement

        optimal_score_box = SurroundingRectangle(
            score_group[best_start : best_end + 1], color=RED
        )
        self.play(Create(optimal_score_box))

        arrow = Arrow(
            start=optimal_score_box.get_bottom(),
            end=optimal_score_box.get_bottom() + DOWN,
            color=RED,
        )
        self.play(Create(arrow))

        explanation = Text(
            f"Optimal score range: chunks {best_start}-{best_end}", font_size=24
        )
        explanation.next_to(arrow, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)
        self.play(
            FadeOut(score_group),
            FadeOut(score_boxes),
            FadeOut(optimal_score_box),
            FadeOut(arrow),
            FadeOut(explanation),
        )

    def show_conclusion(self):
        conclusion = (
            VGroup(
                Text("Optimal Chunking Strategy 🎯", font_size=36),
                Text(
                    "1. 📈 Identify the 'sweet spot' with highest improvement rates",
                    font_size=24,
                ),
                Text(
                    "2. 🔍 Prioritize chunks in both optimal improvement and high-score ranges",
                    font_size=24,
                ),
                Text(
                    "3. 🔬 Expand chunk sizes in low-improvement and peak-score areas",
                    font_size=24,
                ),
                Text(
                    "4. ⚖️ Balance rapid early gains with targeted later refinements",
                    font_size=24,
                ),
                Text(
                    "5. 🔄 Iteratively reassess ranges throughout the summarization process",
                    font_size=24,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.5)
            .to_edge(LEFT, buff=1)
            .shift(UP * 0.5)
        )

        self.play(Write(conclusion))
        self.wait(3)
        self.play(FadeOut(conclusion))
