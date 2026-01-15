def generate_study_plan(pred, f):
    if pred == 0:
        return {
            "Daily Hours": 4,
            "Weekly Hours": 20,
            "Recommendations": [
                "Revise basics daily",
                "Practice MCQs",
                "Watch topic videos"
            ],
            "Weak Areas to Focus": ["Math", "Programming"]
        }
    elif pred == 1:
        return {
            "Daily Hours": 2,
            "Weekly Hours": 10,
            "Recommendations": [
                "Solve past papers",
                "Improve weak areas"
            ],
            "Weak Areas to Focus": ["Database", "DSA"]
        }
    else:
        return {
            "Daily Hours": 1,
            "Weekly Hours": 5,
            "Recommendations": [
                "Learn advanced topics",
                "Start small projects"
            ],
            "Weak Areas to Focus": ["Advanced DSA", "ML"]
        }
