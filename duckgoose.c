#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct Person{
    int id;
    int speed;
    struct Person* next;
}Person;

Person updatePerson(int id, int speed){
    Person np;
    np.id = id;
    np.speed = speed;
    np.next = NULL;

    return np;  
}

void addPerson(Person** start, Person* new) {
    if (*start == NULL) {  // If the list is empty
        *start = new;      // Initialize the list with the new person
        new->next = new;   // Point to itself (circular linked list)
    } else {
        Person* temp = *start;
        while (temp->next != *start) {
            temp = temp->next;  // Traverse the list to the last person
        }
        temp->next = new;  // Link the last person to the new person
        new->next = *start;  // Point the new person to the start
    }
}

void printList(Person* start) {  //used for debugging to check
    if (start == NULL) {
        printf("List is empty.\n");
        return;
    }

    Person* temp = start;
    printf("Current List: ");
    do {
        printf("%d ", temp->id);
        temp = temp->next;
    } while (temp != start);

    printf("\n");
}


// Returns the winner
Person* play(Person* start) {
    while (1) {
        if (start->next == start) {
            printf("winner! %d\n", start->id);
            return start;
        }

        int count = start->id;
        Person* nextP = start->next;

        for (int i = 0; i < count; i++) {
            if (nextP != start){
                printf("%d duck %d\n", start->id, nextP->id);
            }else i -= 1;
            nextP = nextP->next;
        }
        if (nextP == start) nextP = nextP->next;

        printf("%d goose! %d\n", start->id, nextP->id);

        if (start->speed > nextP->speed) {
            Person* temp = nextP;
            while (temp->next != nextP) {
                temp = temp->next;
            }
            temp->next = nextP->next;
            free(nextP);
        } else {
            Person* temp = start;
            while (temp->next != start) {
                temp = temp->next;
            }
            temp->next = start->next;
            Person* oldStart = start;
            start = nextP;
            free(oldStart);
        }
    }
}

int main(){
    int p, s;

    // List of people
    Person* lop = NULL;

    // Read in participants
    while (scanf("%d %d", &p, &s) == 2){
        Person* np = (Person*)malloc(sizeof(Person));
        *np = updatePerson(p, s);
        addPerson(&lop, np);
    }

    Person* winner = play(lop);
    free(winner);
    
    return 0;
}