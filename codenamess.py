import random
import pygame
import gensim.downloader as gd
import sys
import numpy as np
import time
import itertools

class codenames():
    # avail_words = list of words to choose from
    # board size = number of rows/columns
    # death_word_present = boolean for whether or not a "death word" exists on the booard
    # mode = whether each spymaster is choosing random or educated clues, ie. 're' = random player 1, educated player 2
    # spy_model = preset to glove-wiki-gigaword-100
    # guess_model = preset to word2vec-google-news-300
    # sim = boolean for whether or not the object is a simulation or a pygame game
    # is_clone = boolean for whether or not the object is a clone (used in alpha beta)
    def __init__(self, avail_words=None, board_size=5, death_word_present = True, mode = "rr", spy_model = None, guess_model = None, sim = True, is_clone = False, plies = 2):

        # initialize board_dict data struct, input variables, current player
        self.board_size = board_size
        self.board_dict = {}
        self.death_word_present = death_word_present
        self.mode = mode # "r" for random "e" for educated
        self.current_player = 1

        # set models
        if spy_model == None:
            print("Loading spy model...")
            self.spy_model = gd.load("glove-wiki-gigaword-100")
        else: 
            self.spy_model = spy_model

        if guess_model == None:
            print("Loading guess model...")
            self.guess_model = gd.load("word2vec-google-news-300")
        else:
            self.guess_model = guess_model

        # if this game is not a simulated game, take out words not in either model. This is done in the simulate_games function for simulated games to save time
        if sim == False:
            self.avail_words = [word for word in avail_words if word in self.spy_model and word in self.guess_model]
            #print(f"Loaded {len(self.avail_words)} words") 
        else:
            self.avail_words = avail_words


        # initialize word categories
        self.words = []
        self.p1_words = []
        self.p2_words = []
        self.neutral_words = []
        self.death_word = ''

        # initialize scores, list of guessed/unguessed
        self.p1_score = 0
        self.p2_score = 0
        self.moves = 0
        self.p1_guessed = []
        self.p2_guessed = []
        self.neutral_death_guessed = []
        self.unguessed = []
        self.sim = sim
        self.plies = plies

        # checks who hits the death word
        self.death_word_hit = 0

        if is_clone == False:
            self.create_board() 



    # sets up the board_dict data structure
    def create_board(self):
        # randomly assign classes to words; 0=p1, 1=p2, 2=neutral, 3=death word
        word_num = self.board_size**2
        word_num_per_team = word_num // 3 
        neutral_words = word_num - 2 * word_num_per_team - 1 if self.death_word_present else word_num - 2 * word_num_per_team

        # p1 has one extra word, as in codenames, to combat an advantage. In this, instead the players get the same number of guesses and each team has the same number of words
        word_assignments = [0] * word_num_per_team + [1] * word_num_per_team + [2] * neutral_words + [3] if self.death_word_present else [0] * word_num_per_team + [1] * word_num_per_team + [2] * neutral_words
        random.shuffle(word_assignments)

        self.words = random.sample(self.avail_words,word_num)
        self.unguessed = list(self.words)
        
        # create the board dictionary, and fill the player/neutral word lists
        # board dictionary has the form (x,y): [word, label] where x and y are the columns on the board and label is which team the word belongs to 
        idx = 0
        for y in range(self.board_size):
            for x in range(self.board_size):
                label = word_assignments[idx]
                word = self.words[idx]
                self.board_dict[(x,y)] = [word,label]

                if label == 0:
                    self.p1_words.append(word)
                elif label == 1:
                    self.p2_words.append(word)
                elif label == 2:
                    self.neutral_words.append(word)
                else:
                    self.death_word = word

                idx += 1



    # clone method used for alpha beta
    def clone(self):
        new_game = codenames(is_clone=True, spy_model = self.spy_model, guess_model = self.guess_model, board_size = self.board_size, death_word_present=self.death_word_present, mode=self.mode)
        new_game.p1_score = self.p1_score
        new_game.p2_score = self.p2_score
        new_game.p1_words = self.p1_words
        new_game.p2_words = self.p2_words
        new_game.p1_guessed = list(self.p1_guessed)
        new_game.p2_guessed = list(self.p2_guessed)

        new_game.neutral_words = self.neutral_words
        new_game.neutral_death_guessed = list(self.neutral_death_guessed)
        new_game.death_word = self.death_word

        new_game.unguessed = list(self.unguessed)
        new_game.words = self.words

        new_game.plies = self.plies

        new_game.current_player = self.current_player

        return new_game
    


    # play the game with pygame
    def play_game(self):
        pygame.init()

        # secret is false when the word labels are hidden, true when the labels are marked with colors
        secret = False

        # input_tracker is true while the game is running
        input_string = ""
        input_tracker = True
    
        # set up window size
        grid_width = self.board_size
        grid_height = self.board_size
        cell_size = 600/self.board_size
        window_size = (grid_width * cell_size, grid_height * cell_size + 120)
        screen = pygame.display.set_mode(window_size)

        # set colors and fonts
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        blue = (0, 0, 255)
        green = (0, 255, 0)
        beige = (145, 145, 120)
        font1 = pygame.font.SysFont('Arial', 70//self.board_size)
        font2 = pygame.font.SysFont('Arial',20)
        font3 = pygame.font.SysFont('Arial',15)
        font4 = pygame.font.SysFont('Arial',12)

        # get first clue, set running
        tab_tracker = 0
        running = True

        clue_type = self.mode[self.current_player - 1]
        clue = self.get_random_clue(random.randint(1,3)) if clue_type == "r" else self.get_educated_clue()[1]

        while running:

            # check if the game is over, or if the game ends before player gets a makeup round
            if self.winning_eval():
                input_tracker = False
                screen.fill(black)
                secret = True

            for event in pygame.event.get():
                # quit if window closed
                if event.type == pygame.QUIT:
                    running = False
                
                # toggle tab to reveal word categories for testing
                elif event.type == pygame.KEYDOWN and input_tracker:
                    if event.key == pygame.K_TAB:
                        if tab_tracker % 2 == 0:
                            secret = True
                        else:
                            secret = False
                        tab_tracker += 1

                    # take other inputs as input string
                    else:
                        if event.key == pygame.K_BACKSPACE:
                            input_string = input_string[:-1]
                            screen.fill(black)
                        elif event.key == pygame.K_RETURN:
                            guesses = input_string.split()

                            # check if guesses are valid
                            invalid_guesses = 0
                            for guess in guesses:
                                if guess not in self.unguessed:
                                    invalid_guesses += 1
                            
                            # check if any invalid guesses, or if there are more guesses than the clue describes
                            if invalid_guesses > 0 or len(guesses) > clue[1]:
                                wrong_text = font2.render("Invalid guesses", True, white)
                                screen.blit(wrong_text, (window_size[0] // 3, window_size[1] - 20))
                                break

                            # if valid guesses, evaluate guesses
                            self.eval_guesses(guesses)
                            self.current_player = self.current_player % 2 + 1
                            self.moves += 1

                            # get new clue
                            clue_type = self.mode[self.current_player - 1]
                            clue = self.get_random_clue(random.randint(1,3)) if clue_type == "r" else self.get_educated_clue()[1]

                            screen.fill(black)
                            input_string = ""

                        else:
                            input_string += event.unicode

                    text = font2.render(input_string, True, white)
                    screen.blit(text, (window_size[0] // 3, window_size[1] - 50))

            # fill the grid
            for (x, y), word in self.board_dict.items():
                pygame.draw.rect(screen, white, (x * cell_size, y * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, black, (x * cell_size, y * cell_size, cell_size, cell_size), 2)

                # black text until each word is guessed, or if tab is pressed
                if secret == False and word[0] in self.unguessed:
                    word_text = font1.render(word[0], True, black) 
                else:
                    # team 1 is blue, team 2 is red, neutral is beige, death word is green
                    if word[1] == 0:
                        word_text = font1.render(word[0], True, blue)
                    elif word[1] == 1:
                        word_text = font1.render(word[0], True, red)
                    elif word[1] == 2:
                        word_text = font1.render(word[0], True, beige)
                    else:
                        word_text = font1.render(word[0],True, green)

                screen.blit(word_text, (x * cell_size + 15, y * cell_size + 35))

            # print everything
            score = font2.render(f"Score: {self.p1_score} - {self.p2_score}", True, white)
            player = font2.render(f"Team {self.current_player} turn", True, white)
            if clue:
                clue_string = font2.render(f"Clue: {clue[0]} for {clue[1]}", True, white)
                screen.blit(clue_string, (window_size[0] // 3, window_size[1] - 80))

            rules1 = font2.render("Rules:", True, white)
            rules2 = font3.render("+1 for correct guess", True, white)
            rules3 = font3.render("-0.5 for neutral guess", True, white)
            rules4 = font3.render("-5 for death word", True, white)
            rules5 = font4.render("Type guesses sep by space", True, white)
            rules6 = font3.render("Type TAB for testing", True, white)

            screen.blit(score, (5, window_size[1] - 110))
            screen.blit(player, (window_size[0] // 3, window_size[1] - 110))
            screen.blit(rules1, (window_size[0] // 3 * 2, window_size[1] - 110))
            screen.blit(rules2, (window_size[0] // 3 * 2, window_size[1] - 80))
            screen.blit(rules3, (window_size[0] // 3 * 2, window_size[1] - 50))
            screen.blit(rules5, (5, window_size[1] - 80))
            screen.blit(rules6, (5, window_size[1] - 50))
            if self.death_word_present:
                screen.blit(rules4, (window_size[0] // 3 * 2, window_size[1] - 20))
            
            # win condition
            if not input_tracker:
                winner = "Tie"
                if self.p1_score > self.p2_score:
                    winner = "P1"
                elif self.p2_score > self.p1_score:
                    winner = "P2"
                
                win_text = font2.render(f"{winner} wins in {self.moves} moves!", True, white) if winner != "Tie" else font2.render("It's a tie!", True, white)
                screen.blit(win_text, (window_size[0] // 3, window_size[1] - 50))

            pygame.display.update()
            
        pygame.quit()
    


    # function to get similarity using the model and cosine similarities
    def get_sim(self, w1, w2, model):
        similarity = model.similarity(w1, w2) # find cosine similarity

        return similarity



    # function to get random clue, returns a list of the form [clue word, number of words in clue, 'correct' words clue is referring to]
    def get_random_clue(self, words_in_clue = 2):
        attempts = 0
        while attempts < 100: # keep trying until successful, but after 100 attempts give up and select a random word
            # pick a random unguessed word on the board
            random_word = random.choice(self.unguessed) 

            # find most similar words on the board to the random word, INDEPENDENT OF WHICH TEAM THEY ARE ON
            similarities = {word: self.get_sim(random_word, word, self.spy_model) for word in self.unguessed}
            chosen_words = sorted(similarities, key=similarities.get, reverse=True)[:words_in_clue]

            # find a new word, not on the board, to use as the clue
            combined_vector = sum(self.spy_model[word] for word in chosen_words) / words_in_clue # average
            clue = self.spy_model.similar_by_vector(combined_vector, topn = 5 + attempts // 10)  
        
            # filter out the words already on the board and make sure clue is in guess model
            for c, _ in clue:
                if c not in self.words and c in self.guess_model:
                    return [c, words_in_clue, chosen_words] # return the best candidate clue
                
            attempts += 1
        
        #pick a random clue not in the available words but in both datasets
        available_words = [word for word in self.spy_model.index_to_key if word not in self.words and word in self.guess_model]
        random_clue = random.choice(available_words)
        return [random_clue, words_in_clue, chosen_words]
    


    # minimax function to get educated clue using alpha beta 
    def get_educated_clue(self, depth = 0, is_max = True, alpha = float('-inf'), beta = float('inf')):
        # look a few plies down, or until the game is over
        if depth == self.plies * 2 or self.winning_eval():
            return self.p1_score - self.p2_score, None

        if is_max:
            best_val = float('-inf')
            best_clue = None
            # loop through clue combinations
            for clue in self.get_clue_combos():
                new_state = self.next_state(clue) # create clone to manipulate future states
                value, _ = new_state.get_educated_clue(depth + 1, False, alpha, beta)

                if value > best_val:
                    best_val = value
                    best_clue = clue

                alpha = max(alpha, best_val)

                if beta <= alpha:
                    break # pruning

            return best_val, best_clue

        else:
            best_val = float('inf')
            best_clue = None
            for clue in self.get_clue_combos():
                new_state = self.next_state(clue)
                value, _ = new_state.get_educated_clue(depth + 1, True, alpha, beta)

                if value < best_val:
                    best_val = value
                    best_clue = clue

                beta = min(beta, best_val)

                if beta <= alpha:
                    break # pruning

            return best_val, best_clue



    # check clue's similarity to team words, opp words, death word to avoid looking down bad paths in the alpha beta
    # I initially implemented this with model.similarity, but not re-vectorizing on every call is much faster
    def get_clue_score (self, clue_vec, team_vecs, opp_vecs, death_vec):

        # get cosine similarities
        team_scores = np.dot(team_vecs, clue_vec)
        opp_scores = np.dot(opp_vecs, clue_vec) if opp_vecs is not None else 0

        death_score = np.dot(death_vec, clue_vec) if death_vec is not None else 0

        team_avg = np.mean(team_scores)
        opp_max = np.max(opp_scores)

        # punsih close proximity to an opponent's word, severely punish close proximity to a death word
        return team_avg - 2*opp_max - 5*death_score
        


    # generate best combinations in a team's wordlist
    # this function randomly samples from the available team words, and then sorts the clues based on a 'pre-checked' score, taking only the best num_considered options
    def get_clue_combos (self, words_in_clue = 2, samples = 30, num_considered = 5):
        avail_team_words = [word for word in self.p1_words if word in self.unguessed] if self.current_player == 1 else [word for word in self.p2_words if word in self.unguessed]
        avail_opp_words = [word for word in self.unguessed if word not in avail_team_words and word not in self.neutral_words]

        if not avail_team_words:
            return []
        
        sampled_combos = set()
        clue_combos = {}

        # calculate team vectors here
        team_vecs = np.array([self.spy_model[word] for word in avail_team_words])
        opp_vecs = np.array([self.spy_model[word] for word in avail_opp_words]) if avail_opp_words else None
        death_vec = self.spy_model[self.death_word] if self.death_word_present else None

        # randomly sample combos to not get extremely time expensive
        for x in range(samples):
            if len(avail_team_words) >= 4: # if there are more words, incentivize bigger clues
                random.choices([2,3,4], weights=[0.5,0.3,0.2], k=1)[0]
            elif len(avail_team_words) == 1: # if there is only 1 word, do a 1 word clue
                words_in_clue = 1
            else:
                words_in_clue = 2 # else do 2 clues

            # sort to avoid permutation duplicates
            combo = tuple(sorted(random.sample(avail_team_words, words_in_clue))) 
            sampled_combos.add(combo)

        for chosen_words in sampled_combos:
            # find a new word, not on the board, to use as the clue
            combined_vector = sum(self.spy_model[word] for word in chosen_words) / len(chosen_words)
            clues = self.spy_model.similar_by_vector(combined_vector, topn=5)

            # filter out the words already on the board and make sure clue is in guess model
            # also 'pre-check' clues to avoid bad pathways for the alpha beta

            best_clue = None
            best_score = float('-inf')
            for c, _ in clues:
                if c not in self.words and c in self.guess_model:
                    clue_vec = self.spy_model[c]
                    score = self.get_clue_score(clue_vec, team_vecs, opp_vecs, death_vec)
                    
                    if score > best_score:
                        best_clue = [c, len(chosen_words), chosen_words]
                        best_score = score
            
            if best_clue:
                clue_combos[best_score] = best_clue
        
        sorted_clue_combos = sorted(clue_combos.items(), key=lambda item: item[0],reverse=True)
        if len(sorted_clue_combos) > num_considered:
            sorted_clue_combos = sorted_clue_combos[:num_considered]
        
        clues_to_return = [item[1] for item in sorted_clue_combos]

        return clues_to_return
    

    # looks at all combinations
    def get_clue_combos2 (self, words_in_clue=2):
        avail_team_words = [word for word in self.p1_words if word in self.unguessed] if self.current_player == 1 else [word for word in self.p2_words if word in self.unguessed]
        if not avail_team_words:
            return []
        
        clue_combos = []

        for chosen_words in list(itertools.combinations(avail_team_words, words_in_clue)):
            # find a new word, not on the board, to use as the clue
            combined_vector = sum(self.spy_model[word] for word in chosen_words) / len(chosen_words)
            clues = self.spy_model.similar_by_vector(combined_vector, topn=5)

            # filter out the words already on the board and make sure clue is in guess model
            for c, _ in clues:
                if c not in self.words and c in self.guess_model:
                    clue_combos.append([c, words_in_clue, list(chosen_words)])
                    break

        # if less than words_in_clue words left, do a simplter clue
        while len(clue_combos) == 0:
            words_in_clue -= 1
            for chosen_words in list(itertools.combinations(avail_team_words, words_in_clue)):

                # find a new word, not on the board, to use as the clue
                combined_vector = sum(self.spy_model[word] for word in chosen_words) / len(chosen_words)
                clues = self.spy_model.similar_by_vector(combined_vector, topn=5)

                # filter out the words already on the board and make sure clue is in guess model
                for c, _ in clues:
                    if c not in self.words and c in self.guess_model:
                        clue_combos.append([c, words_in_clue, list(chosen_words)])
                        break

        return clue_combos
    


    # simulate next_state given a clue, returning the new state
    def next_state(self, clue):
        new_state = self.clone()

        similarities = {word: self.get_sim(clue[0], word, self.guess_model) for word in self.unguessed}
        guesses = sorted(similarities, key=similarities.get, reverse=True)[:clue[1]]

        new_state.eval_guesses(guesses)
        new_state.current_player = new_state.current_player % 2 + 1

        return new_state



    # function to evaluate guesses and update scores
    def eval_guesses(self, guesses):
        for guess in guesses:
            # 1 point to whichever team the guess belongs to, -0.5 points if neutral word, -5 for death word
            if guess in self.p1_words:
                self.p1_score += 1
                self.p1_guessed.append(guess)
            elif guess in self.p2_words:
                self.p2_score += 1
                self.p2_guessed.append(guess)
            elif guess in self.neutral_words:
                if self.current_player == 1:
                    self.p1_score -= 0.5
                else:
                    self.p2_score -= 0.5
                self.neutral_death_guessed.append(guess)
            elif guess == self.death_word:
                self.death_word_hit = self.current_player
                if self.current_player == 1:
                    self.p1_score -= 5
                else:
                    self.p2_score -= 5
            
            self.unguessed.remove(guess)
         


    # check if all words are guessed for a given team
    def winning_eval(self):
        if (len(self.p1_guessed) == len(self.p1_words) or len(self.p2_guessed) == len(self.p2_words)) and self.current_player == 1:
            return True
        

        return False
    


    # simulate a game and return the winner
    def simulate_game(self):
        while True:

            # check for winner but only stop game after player 2's turn or if there are no words in self.unguessed
            if self.winning_eval():
                break

            # edge case where all clues are guessed on player 1's turn
            if not self.unguessed:
                break

            clue_type = self.mode[self.current_player - 1]
            clue = self.get_random_clue(random.randint(1,3)) if clue_type == "r" else self.get_educated_clue()[1]
            
            # get guesses using different model

            if clue:
                similarities = {word: self.get_sim(clue[0], word, self.guess_model) for word in self.unguessed}
                guesses = sorted(similarities, key=similarities.get, reverse=True)[:clue[1]]

                self.eval_guesses(guesses)

            # change player
            self.current_player = self.current_player % 2 + 1
            self.moves += self.current_player % 2
        
        # return winner
        winner = "Tie"
        if self.p1_score > self.p2_score:
            winner = "P1"
        elif self.p2_score > self.p1_score:
            winner = "P2"
        
        return winner, self.moves, (self.p1_score, self.p2_score), self.death_word_hit



# simulate games and return counts, printing percentages
# reps = number of games, board_size = number of rows/columns, word_list = available word list, mode = "rr," "re", "er", or "ee" where e marks a random player and e marks an educated player
def simulate_games(reps, board_size, word_list, mode, death_word_present = False, spy_model = None, guess_model = None, plies=2):

    start = time.time()

    # set models, using preset libraries if nothing is passed in 
    if spy_model == None:
        print("Loading spy model...")
        spy_model = gd.load("glove-wiki-gigaword-100")
    else: 
        spy_model = spy_model

    if guess_model == None:
        print("Loading guess model...")
        guess_model = gd.load("word2vec-google-news-300")
    else:
        guess_model = guess_model

    avail_words = [word for word in word_list if word in spy_model and word in guess_model]
    #print(f"{len(avail_words)} possible words loaded")

    count_dict = {"P1":0, "P2":0, "Tie":0}
    
    # track total moves for average
    moves_tot = 0
    p1_score_tot = 0
    p2_score_tot = 0
    dwh_counts = {1: 0, 2: 0, 0: 0}

    for x in range(reps):
        c = codenames(avail_words=avail_words, board_size=board_size, death_word_present=death_word_present, spy_model=spy_model, guess_model=guess_model, mode=mode, plies=plies)

        winner, moves, score, dwh = c.simulate_game()
        count_dict[winner] += 1
        moves_tot += moves
        p1_score_tot += score[0]
        p2_score_tot += score[1]
        if c.death_word_present:
            dwh_counts[dwh] += 1

        # create a loading bar for bigger tasks
        progress = (x + 1) / reps
        bar_length = 40  
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
        sys.stdout.write(f'\r|{bar}| {int(progress * 100)}%')
        sys.stdout.flush()
    
    # print percentages, average moves per game
    print(f"\nSimulated {reps} games")
    percentages = {x: (n / reps) * 100 for x, n in count_dict.items()}  
    for x, percentage in percentages.items():
        print(f"{x}: {percentage:.2f}%")
    print(f"Average score: {p1_score_tot/reps:.2f} - {p2_score_tot/reps:.2f}")
    
    if death_word_present:
        dwh_perc = {x: (n/reps) * 100 for x, n in dwh_counts.items()}
        for x, perc in dwh_perc.items():
            print(f"P{x} gets death word: {perc:.2f}%") if x in (1,2) else print(f"No one gets death word: {perc:.2f}%") 

    end = time.time()
    print(f"Average moves per team per game: {moves_tot / reps:.2f}")
    print(f"Time simulating: {end-start}")

    return count_dict
        


# function to test all four modes, rr, re, er, and ee
def test_modes(reps, board_size, word_list, death_word_present = False, spy_model = None, guess_model = None, plies = 2):
    if spy_model == None:
        print("Loading spy model...")
        spy_model = gd.load("glove-wiki-gigaword-100")
    else: 
        spy_model = spy_model

    if guess_model == None:
        print("Loading guess model...")
        guess_model = gd.load("word2vec-google-news-300")
    else:
        guess_model = guess_model


    print("\n--------------Testing random-random:------------------")
    simulate_games(reps, board_size, word_list, "rr", spy_model=spy_model, guess_model=guess_model, death_word_present=death_word_present, plies=plies) 
    print("\n--------------Testing random-educated:------------------")
    simulate_games(reps, board_size, word_list, "re", spy_model=spy_model, guess_model=guess_model, death_word_present=death_word_present, plies=plies)
    print("\n--------------Testing educated-random:------------------")
    simulate_games(reps, board_size, word_list, "er", spy_model=spy_model, guess_model=guess_model, death_word_present=death_word_present, plies=plies)
    print("\n--------------Testing educated-educated:------------------")
    simulate_games(reps, board_size, word_list, "ee", spy_model=spy_model, guess_model=guess_model, death_word_present=death_word_present, plies=plies)




# 25 random words for testing
word_list = ["bird","dog","soup","beans","king","queen","music","capital","monument","baseball","frog","door","instrument","law","salad","sun","president","city","town","computer","sandwich","book","pencil","ball","college"]

# 389 nouns extracted from https://github.com/Gullesnuffs/Codenames/blob/master/wordlist-eng.txt
with open('csci 3022/project/codenames_words.txt', 'r') as file:
    codenames_word_list = [line.strip().lower() for line in file.readlines()]



#c = codenames(avail_words=codenames_word_list, board_size=3, death_word_present=True, mode='ee', sim = False)
#c.play_game()

#simulate_games(1, 4, codenames_word_list, "ee", plies=5, death_word_present=True, guess_model=gd.load("glove-wiki-gigaword-100"))

test_modes(100, 3, codenames_word_list, death_word_present=True, plies=5)