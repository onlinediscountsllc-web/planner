"""
ULTIMATE LIFE PLANNER - QUICK START GUIDE
==========================================

This guide shows you how to use all the amazing features of the Enhanced Life Planning System!

TABLE OF CONTENTS:
1. Basic Setup
2. Virtual Pet System
3. Habit Tracking
4. Goal Management
5. Life Balance Wheel
6. Pomodoro Focus Sessions
7. Gamification & Achievements
8. Analytics & Insights
9. Saving & Loading Data
10. Advanced Features
"""

from enhanced_life_planner_ultimate import (
    UltimateLifePlanningSystem,
    PetPersonality,
    EnhancedVirtualPet,
    SMARTGoalValidator,
    LifeBalanceWheel,
    HabitTracker
)
from datetime import datetime, timedelta
import random


# ==============================================================================
# 1. BASIC SETUP
# ==============================================================================

def example_basic_setup():
    """Create a new life planning system"""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC SETUP")
    print("="*70 + "\n")
    
    # Create system with custom pet
    system = UltimateLifePlanningSystem(
        pet_name="Nova",
        pet_species="dragon",  # Options: dragon, phoenix, unicorn, owl, fox, cat
        pet_personality=PetPersonality.ENERGETIC  # ENERGETIC, CALM, CURIOUS, LOYAL, PLAYFUL
    )
    
    print(system.pet.get_status_display())
    
    return system


# ==============================================================================
# 2. VIRTUAL PET SYSTEM
# ==============================================================================

def example_pet_interactions(system):
    """Demonstrate all pet interactions"""
    print("\n" + "="*70)
    print("EXAMPLE 2: VIRTUAL PET INTERACTIONS")
    print("="*70 + "\n")
    
    # Feed your pet
    print("--- Feeding Pet ---")
    msg, xp = system.pet.feed(food_quality=80)
    print(msg)
    print(f"XP earned: {xp}\n")
    
    # Play with pet
    print("--- Playing with Pet ---")
    activities = ['fetch', 'hide_seek', 'training', 'puzzle']
    msg, xp = system.pet.play(random.choice(activities))
    print(msg)
    print(f"XP earned: {xp}\n")
    
    # Train pet stats
    print("--- Training Pet ---")
    stats = ['intelligence', 'strength', 'charisma', 'wisdom']
    msg, xp = system.pet.train(random.choice(stats))
    print(msg)
    print(f"XP earned: {xp}\n")
    
    # Use special ability
    print("--- Using Special Ability ---")
    msg, effects = system.pet.use_special_ability()
    print(msg)
    print(f"Effects: {effects}\n")
    
    # Play mini-game
    print("--- Mini-Game ---")
    game_score = random.randint(50, 150)
    msg, xp = system.pet.play_mini_game("memory_match", game_score)
    print(msg)
    print(f"XP earned: {xp}\n")
    
    # Let pet rest
    print("--- Rest Time ---")
    msg = system.pet.rest()
    print(msg + "\n")
    
    # Show updated status
    print(system.pet.get_status_display())


# ==============================================================================
# 3. HABIT TRACKING
# ==============================================================================

def example_habit_tracking(system):
    """Demonstrate habit tracking with science-based formation curves"""
    print("\n" + "="*70)
    print("EXAMPLE 3: HABIT TRACKING")
    print("="*70 + "\n")
    
    # Add habits
    print("--- Adding Habits ---")
    habits = [
        ("Morning Meditation", 66),
        ("Daily Exercise", 66),
        ("Reading 30 min", 66),
        ("Drink 8 Glasses Water", 21),
        ("Journal Before Bed", 66)
    ]
    
    for habit_name, days in habits:
        msg = system.add_habit(habit_name, days)
        print(msg)
    print()
    
    # Complete habits
    print("--- Completing Habits ---")
    for habit_name, _ in habits[:3]:
        msg = system.complete_habit(habit_name)
        print(msg)
    print()
    
    # Show habit statistics
    print("--- Habit Statistics ---")
    for habit_name, habit in system.habits.items():
        stats = habit.get_stats()
        print(f"\n{habit_name}:")
        print(f"  Formation: {stats['formation_percentage']:.1f}%")
        print(f"  Current Streak: {stats['current_streak']} days")
        print(f"  Longest Streak: {stats['longest_streak']} days")
        print(f"  Completion Rate: {stats['completion_rate']:.1f}%")
    print()


# ==============================================================================
# 4. GOAL MANAGEMENT
# ==============================================================================

def example_goal_management(system):
    """Demonstrate SMART goal validation and tracking"""
    print("\n" + "="*70)
    print("EXAMPLE 4: GOAL MANAGEMENT")
    print("="*70 + "\n")
    
    # Add goals with SMART validation
    print("--- Adding Goals ---\n")
    
    goals_to_add = [
        {
            'title': 'Complete Python Course',
            'description': 'Finish the Advanced Python course on Coursera with 90% or higher, completing all 12 modules and 50 coding exercises by March 31st, 2026',
            'deadline': datetime(2026, 3, 31),
            'category': 'learning',
            'metrics': ['modules_completed', 'exercises_done', 'final_score']
        },
        {
            'title': 'Improve Health',
            'description': 'Exercise 4 times per week, lose 10 pounds, and run a 5K by June 2026',
            'deadline': datetime(2026, 6, 30),
            'category': 'health',
            'metrics': ['weight', 'workouts_per_week', '5k_time']
        },
        {
            'title': 'Save Money',
            'description': 'Save $5,000 for emergency fund by December 2026 through monthly contributions of $500',
            'deadline': datetime(2026, 12, 31),
            'category': 'finances',
            'metrics': ['total_saved', 'monthly_contribution']
        }
    ]
    
    for goal_data in goals_to_add:
        result = system.add_goal(**goal_data)
        print(f"Goal: {goal_data['title']}")
        print(f"  SMART Score: {result['validation']['overall']:.1f}/100")
        print(f"  Grade: {result['validation']['grade']}")
        
        if result['validation']['feedback']:
            print("  Feedback:")
            for feedback in result['validation']['feedback']:
                print(f"    - {feedback}")
        print()
    
    # Update goal progress
    print("--- Updating Goal Progress ---")
    print(system.update_goal_progress(0, 25))
    print(system.update_goal_progress(1, 40))
    print()


# ==============================================================================
# 5. LIFE BALANCE WHEEL
# ==============================================================================

def example_life_balance(system):
    """Demonstrate life balance tracking"""
    print("\n" + "="*70)
    print("EXAMPLE 5: LIFE BALANCE WHEEL")
    print("="*70 + "\n")
    
    # Update different dimensions
    print("--- Updating Life Dimensions ---")
    updates = [
        ('health', 7.5),
        ('career', 8.0),
        ('finances', 6.0),
        ('relationships', 7.0),
        ('personal_growth', 8.5),
        ('recreation', 5.5),
        ('environment', 6.5),
        ('spirituality', 7.0)
    ]
    
    for dimension, score in updates:
        msg = system.life_balance.update_dimension(dimension, score)
        print(msg)
    print()
    
    # Show balance visualization
    print(system.life_balance.visualize_text())
    print()
    
    # Get balance score
    balance_score = system.life_balance.calculate_balance_score()
    print(f"Overall Balance Score: {balance_score:.1f}/100\n")


# ==============================================================================
# 6. POMODORO FOCUS SESSIONS
# ==============================================================================

def example_pomodoro(system):
    """Demonstrate Pomodoro technique integration"""
    print("\n" + "="*70)
    print("EXAMPLE 6: POMODORO FOCUS SESSIONS")
    print("="*70 + "\n")
    
    # Start a Pomodoro session
    print("--- Starting Pomodoro ---")
    msg = system.start_pomodoro("Deep Work on Python Project", 25)
    print(msg)
    print()
    
    # Complete session
    print("--- Completing Pomodoro ---")
    msg, xp = system.complete_pomodoro()
    print(msg)
    print(f"XP earned: {xp}\n")
    
    # Start another session
    system.start_pomodoro("Review Code", 25)
    system.complete_pomodoro()
    
    # Get statistics
    print("--- Pomodoro Statistics ---")
    stats = system.pomodoro.get_statistics()
    print(f"Total Sessions: {stats['total_sessions']}")
    print(f"Total Focus Time: {stats['total_focus_time']} minutes")
    print(f"Sessions Today: {stats['sessions_today']}")
    print(f"Average Interruptions: {stats['average_interruptions']:.2f}\n")


# ==============================================================================
# 7. GAMIFICATION & ACHIEVEMENTS
# ==============================================================================

def example_gamification(system):
    """Demonstrate gamification features"""
    print("\n" + "="*70)
    print("EXAMPLE 7: GAMIFICATION & ACHIEVEMENTS")
    print("="*70 + "\n")
    
    # Show gamification progress
    print(system.gamification.get_progress_summary())
    print()
    
    # Unlock some achievements manually for demo
    achievements = [
        'tasks_10',
        'goals_5',
        'pet_bond_100',
        'focus_10h'
    ]
    
    print("--- Unlocking Achievements ---")
    for ach_id in achievements:
        unlocked, points = system.gamification.unlock_achievement(ach_id)
        if unlocked:
            ach = system.gamification.achievements[ach_id]
            print(f"{ach.icon} {ach.name} unlocked! +{points} points")
    print()


# ==============================================================================
# 8. ANALYTICS & INSIGHTS
# ==============================================================================

def example_analytics(system):
    """Demonstrate advanced analytics and pattern detection"""
    print("\n" + "="*70)
    print("EXAMPLE 8: ANALYTICS & INSIGHTS")
    print("="*70 + "\n")
    
    # Log some daily data to build history
    print("--- Logging Daily Data ---")
    for day in range(14):
        mood = 60 + random.randint(-15, 25)
        stress = 45 + random.randint(-20, 20)
        productivity = 65 + random.randint(-20, 25)
        
        result = system.log_daily_data({
            'mood': mood,
            'stress': stress,
            'energy': 70 + random.randint(-20, 20),
            'productivity': productivity,
            'sleep_hours': 7 + random.randint(-2, 2),
            'exercise_minutes': random.randint(0, 60),
            'social_time_minutes': random.randint(30, 180)
        })
        print(f"Day {day + 1}: Mood={mood:.0f}, Stress={stress:.0f}, Productivity={productivity:.0f}")
    print()
    
    # Get personalized recommendations
    print("--- Personalized Recommendations ---")
    recommendations = system.get_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print()
    
    # Pattern detection
    print("--- Pattern Detection ---")
    
    # Circadian patterns
    circadian = system.pattern_detector.detect_circadian_pattern()
    if circadian.get('pattern_found'):
        print(f"\n⏰ Best time of day: {circadian['best_hour']}:00")
        print(f"   Average mood: {circadian['best_hour_mood']:.1f}")
        print(f"   {circadian['recommendation']}")
    
    # Correlations
    correlations = system.pattern_detector.detect_correlations()
    if correlations:
        print("\n📊 Activity-Mood Correlations:")
        for corr in correlations[:3]:
            if 'activity' in corr:
                print(f"   {corr['activity']}: {corr['correlation']:.2f} ({corr['strength']} {corr['impact']})")
                print(f"   → {corr['recommendation']}")
    
    # Mood forecast
    forecast = system.pattern_detector.forecast_mood(7)
    if forecast:
        print("\n🔮 7-Day Mood Forecast:")
        for day, mood in enumerate(forecast, 1):
            emoji = "😊" if mood > 70 else "😐" if mood > 50 else "😔"
            print(f"   Day {day}: {mood:.0f} {emoji}")
    print()


# ==============================================================================
# 9. SAVING & LOADING DATA
# ==============================================================================

def example_save_load(system):
    """Demonstrate data persistence"""
    print("\n" + "="*70)
    print("EXAMPLE 9: SAVING & LOADING DATA")
    print("="*70 + "\n")
    
    # Save data
    print("--- Saving Data ---")
    msg = system.save_to_file("my_life_planner.json")
    print(msg)
    print()
    
    # Load data
    print("--- Loading Data ---")
    loaded_system = UltimateLifePlanningSystem.load_from_file("my_life_planner.json")
    print(f"✅ Loaded system with pet '{loaded_system.pet.name}' at level {loaded_system.pet.stats.level}")
    print(f"   Habits tracked: {len(loaded_system.habits)}")
    print(f"   Goals: {len(loaded_system.goals)}")
    print(f"   Gamification level: {loaded_system.gamification.level}")
    print()


# ==============================================================================
# 10. ADVANCED FEATURES
# ==============================================================================

def example_advanced_features(system):
    """Demonstrate advanced mathematical and analytical features"""
    print("\n" + "="*70)
    print("EXAMPLE 10: ADVANCED FEATURES")
    print("="*70 + "\n")
    
    # Mathematical utilities
    print("--- Mathematical Sequences ---")
    from enhanced_life_planner_ultimate import AdvancedMathUtil
    
    print("Golden Ratio:", AdvancedMathUtil.golden_ratio())
    print("Fibonacci(10):", AdvancedMathUtil.fibonacci_sequence(10))
    print("Lucas(10):", AdvancedMathUtil.lucas_sequence(10))
    print()
    
    # Exponential smoothing on mood data
    if len(system.mood_history) >= 7:
        print("--- Time Series Analysis ---")
        smoothed = AdvancedMathUtil.exponential_smoothing(system.mood_history[-14:])
        print(f"Original mood data: {[f'{m:.1f}' for m in system.mood_history[-7:]]}")
        print(f"Smoothed forecast: {[f'{m:.1f}' for m in smoothed[-7:]]}")
        print()
    
    # Moving average
    if len(system.productivity_history) >= 7:
        print("--- Productivity Moving Average ---")
        ma = AdvancedMathUtil.moving_average(system.productivity_history, window=7)
        print(f"7-day moving average: {[f'{m:.1f}' for m in ma[-7:]]}")
        print()
    
    # Entropy calculation
    if len(system.mood_history) >= 10:
        print("--- Mood Entropy (Stability) ---")
        entropy = AdvancedMathUtil.calculate_entropy(system.mood_history)
        print(f"Entropy: {entropy:.2f}")
        print(f"Interpretation: {'Low' if entropy < 2 else 'Medium' if entropy < 3 else 'High'} variability")
        print()
    
    # Markov chain prediction
    if len(system.mood_history) >= 5:
        print("--- Markov Chain Mood Prediction ---")
        current_mood = system.mood_history[-1]
        current_state = system._discretize_mood(current_mood)
        predicted, prob = system.markov_mood.predict_next(current_state)
        print(f"Current mood state: {current_state}")
        print(f"Predicted next state: {predicted} (probability: {prob:.2f})")
        
        # Predict sequence
        sequence = system.markov_mood.predict_sequence(current_state, 5)
        print(f"Predicted 5-day sequence: {sequence}")
        print()
    
    # Bayesian inference
    print("--- Bayesian Activity Recommendation ---")
    # Simulate current state
    evidence = 'high_stress'
    best_activity, confidence = system.bayesian_engine.get_best_hypothesis(evidence)
    print(f"Given evidence: {evidence}")
    print(f"Recommended activity: {best_activity} (confidence: {confidence*100:.1f}%)")
    print()


# ==============================================================================
# COMPREHENSIVE DEMO
# ==============================================================================

def run_complete_demo():
    """Run all examples in sequence"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " ULTIMATE LIFE PLANNING SYSTEM - COMPLETE DEMONSTRATION ".center(68) + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Create system
    system = example_basic_setup()
    
    # Run all examples
    example_pet_interactions(system)
    example_habit_tracking(system)
    example_goal_management(system)
    example_life_balance(system)
    example_pomodoro(system)
    example_gamification(system)
    example_analytics(system)
    example_save_load(system)
    example_advanced_features(system)
    
    # Final dashboard
    print("\n" + "="*70)
    print("FINAL DASHBOARD")
    print("="*70 + "\n")
    print(system.get_analytics_dashboard())
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " DEMONSTRATION COMPLETE! ".center(68) + "║")
    print("╚" + "="*68 + "╝\n")
    
    return system


# ==============================================================================
# QUICK START TEMPLATES
# ==============================================================================

def quick_start_template():
    """A simple template to get started quickly"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  QUICK START TEMPLATE
╠══════════════════════════════════════════════════════════════════╣
║  Copy and customize this code to get started quickly:
╚══════════════════════════════════════════════════════════════════╝

from enhanced_life_planner_ultimate import (
    UltimateLifePlanningSystem, 
    PetPersonality
)
from datetime import datetime

# 1. Create your system
system = UltimateLifePlanningSystem(
    pet_name="MyPet",
    pet_species="dragon",
    pet_personality=PetPersonality.CURIOUS
)

# 2. Add habits you want to track
system.add_habit("Exercise", 66)
system.add_habit("Meditation", 66)
system.add_habit("Reading", 66)

# 3. Add a goal
system.add_goal(
    title="Learn New Skill",
    description="Master a new skill with measurable progress by [date]",
    deadline=datetime(2026, 12, 31),
    category="learning",
    metrics=["progress_percentage"]
)

# 4. Daily routine
def daily_routine():
    # Log your daily data
    system.log_daily_data({
        'mood': 75,
        'stress': 40,
        'energy': 70,
        'productivity': 80,
        'sleep_hours': 7,
        'exercise_minutes': 30,
        'social_time_minutes': 60
    })
    
    # Complete habits
    system.complete_habit("Exercise")
    system.complete_habit("Meditation")
    
    # Do Pomodoro sessions
    system.start_pomodoro("Deep Work", 25)
    # ... work ...
    system.complete_pomodoro()
    
    # Interact with pet
    system.pet.feed(75)
    system.pet.play("training")
    
    # Update life balance
    system.life_balance.update_dimension("health", 8.0)
    
    # Get recommendations
    recommendations = system.get_recommendations()
    for rec in recommendations:
        print(rec)

# 5. Save your progress
system.save_to_file("my_data.json")

# 6. Load later
# system = UltimateLifePlanningSystem.load_from_file("my_data.json")
""")


if __name__ == "__main__":
    # Run the complete demonstration
    system = run_complete_demo()
    
    # Show quick start template
    quick_start_template()
    
    print("\n✨ Try customizing and running this code for your own life planning journey! ✨\n")
