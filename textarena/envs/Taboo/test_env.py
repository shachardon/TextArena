import pytest

import textarena as ta
from textarena.envs.Taboo.env import TabooEnv

CLUE = "a helpful clue zzq"
WRONG_GUESS = "[zzzwrong]"
MAX_STEPS = 2000


def make_env(num_players=2, max_rounds=4, attempts=6, deck=None):
    env = TabooEnv(categories=["things"], max_rounds=max_rounds, max_attempts_per_player=attempts)
    if deck is not None:
        env.data = deck
    env.reset(num_players=num_players, seed=1)
    return env


def small_deck(size):
    return {f"word{i}": [f"taboo{i}a", f"taboo{i}b"] for i in range(size)}


def current_word(env):
    return env.state.game_state["word_to_guess"]


def is_clue_giver(env):
    return env.state.role_mapping[env.state.current_player_id] == "Clue Giver"


def play(env, guess_policy):
    """Play to the end: fixed clues, with guess_policy(env) producing each guess."""
    for _ in range(MAX_STEPS):
        done, _ = env.step(CLUE if is_clue_giver(env) else guess_policy(env))
        if done:
            return env.state.rewards
    pytest.fail("The game did not terminate")


def correct(env):
    return f"[{current_word(env)}]"


def always_wrong(env):
    return WRONG_GUESS


# --- Setup ---

def test_two_players_form_one_pair_with_both_roles():
    env = make_env()
    assert env.cooperative
    assert env.state.role_mapping[0] == "Clue Giver"
    assert env.state.role_mapping[1] == "Guesser"
    assert set(env.state.game_state["score"]) == {0}


@pytest.mark.parametrize("num_players", [1, 3, 5])
def test_unsupported_player_counts_are_rejected(num_players):
    with pytest.raises(AssertionError):
        make_env(num_players=num_players)


def test_prompts_describe_a_shared_goal_without_teams():
    env = make_env()
    for player_id in (0, 1):
        prompt = env._generate_player_prompt(player_id, env.state.game_state)
        assert "share one score" in prompt
        assert "Team" not in prompt


# --- Turn flow ---

def test_players_alternate_and_guesser_sees_the_clue():
    env = make_env()
    assert env.state.current_player_id == 0
    env.step(CLUE)
    assert env.state.current_player_id == 1
    assert CLUE in str(env.state.observations[1])
    env.step(WRONG_GUESS)
    assert env.state.current_player_id == 0


def test_guesser_is_not_told_the_word():
    env = make_env()
    word = current_word(env)
    assert word not in str(env.state.observations[1])
    assert word in str(env.state.observations[0])


def test_game_lasts_rounds_times_exchanges():
    env = make_env(max_rounds=4, attempts=6)
    play(env, always_wrong)
    assert env.state.turn == 48


# --- Shared score ---

def test_both_players_receive_the_shared_score():
    env = make_env()
    guesses = {"made": 0}

    def three_correct_then_wrong(game):
        guesses["made"] += 1
        return correct(game) if guesses["made"] <= 3 else WRONG_GUESS

    assert play(env, three_correct_then_wrong) == {0: 3, 1: 3}


def test_zero_score_gives_both_players_zero():
    env = make_env()
    assert play(env, always_wrong) == {0: 0, 1: 0}


def test_perfect_game_scores_every_exchange():
    env = make_env(max_rounds=4, attempts=6)
    assert play(env, correct) == {0: 24, 1: 24}


# --- Running out of words ---

def test_game_ends_when_correct_guesses_empty_the_deck():
    env = make_env(deck=small_deck(3))
    assert play(env, correct) == {0: 3, 1: 3}
    assert env.state.turn == 6


def test_game_ends_when_a_taboo_violation_empties_the_deck():
    env = make_env(deck=small_deck(1))
    env.step(f"it is {current_word(env)}")
    done, _ = env.step(f"it is {current_word(env)}")
    assert done
    assert env.state.rewards == {0: 0, 1: 0}


# --- Invalid moves ---

def test_repeated_taboo_violation_skips_the_word_and_ends_the_round():
    env = make_env()
    first_word = current_word(env)
    env.step(f"it is {first_word}")
    assert env.state.game_state["round"] == 1  # first offence is only a warning
    assert env.state.current_player_id == 0
    env.step(f"it is {first_word}")
    assert env.state.game_state["round"] == 2
    assert current_word(env) != first_word
    assert env.state.current_player_id == 0
    assert env.state.game_state["score"][0] == 0


@pytest.mark.parametrize("num_players", [2, 4])
def test_repeated_missing_brackets_counts_as_a_wrong_guess(num_players):
    env = make_env(num_players=num_players)
    env.step(CLUE)
    guesser = env.state.current_player_id
    env.step("no brackets")
    assert env.state.current_player_id == guesser  # one retry is allowed
    env.step("still no brackets")
    assert env.state.current_player_id == guesser - 1  # back to the Clue Giver
    assert env.state.game_state["score"][0] == 0


def test_guesser_who_never_uses_brackets_cannot_stall_the_game():
    env = make_env()
    assert play(env, lambda game: "no brackets") == {0: 0, 1: 0}


# --- Guess parsing ---

@pytest.mark.parametrize("template", ["[{}]", "['{}']", '["{}"]', "[ {} ]", "I think it is [{}]!"])
def test_accepted_guess_formats(template):
    env = make_env()
    env.step(CLUE)
    env.step(template.format(current_word(env).upper()))
    assert env.state.game_state["score"][0] == 1


# --- Team mode is unchanged ---

def test_four_players_still_compete_as_two_teams():
    env = make_env(num_players=4)
    assert not env.cooperative
    assert set(env.state.game_state["score"]) == {0, 1}

    def only_team_zero_scores(game):
        return correct(game) if game.state.game_state["current_team"] == 0 else WRONG_GUESS

    assert play(env, only_team_zero_scores) == {0: 1, 1: 1, 2: -1, 3: -1}


# --- Registered environment ---

def test_registered_env_runs_with_two_players():
    env = ta.make("Taboo-v0")
    env.reset(num_players=2, seed=1)
    player_id, observation = env.get_observation()
    assert player_id == 0
    assert "share one score" in observation
    done, _ = env.step(CLUE)
    assert not done
