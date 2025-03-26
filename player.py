import pygame
class Player:
    def __init__(self, x=450, y=663):
        self.x = x
        self.y = y
        self.direction = 0
        self.speed = 2
        self.images = [pygame.transform.scale(pygame.image.load(f'assets/player_images/{i}.png'), (45, 45)) 
                      for i in range(1, 5)]
        self.lives = 3
        self.score = 0

    def draw(self, screen, counter):
        if self.direction == 0:
            screen.blit(self.images[counter // 5], (self.x, self.y))
        elif self.direction == 1:
            screen.blit(pygame.transform.flip(self.images[counter // 5], True, False), (self.x, self.y))
        elif self.direction == 2:
            screen.blit(pygame.transform.rotate(self.images[counter // 5], 90), (self.x, self.y))
        elif self.direction == 3:
            screen.blit(pygame.transform.rotate(self.images[counter // 5], 270), (self.x, self.y))

    def move(self, turns_allowed):
        if self.direction == 0 and turns_allowed[0]:
            self.x += self.speed
        elif self.direction == 1 and turns_allowed[1]:
            self.x -= self.speed
        elif self.direction == 2 and turns_allowed[2]:
            self.y -= self.speed
        elif self.direction == 3 and turns_allowed[3]:
            self.y += self.speed
        if self.x > 900:
            self.x = -47
        elif self.x < -50:
            self.x = 897

    def get_center(self):
        return self.x + 23, self.y + 24