import random

# 課程資料
courses = [
    {'teacher': '  ', 'name': '　　', 'hours': -1},  # 那一節沒上課
    {'teacher': '甲', 'name': '機率', 'hours': 2},
    {'teacher': '甲', 'name': '線代', 'hours': 3},
    {'teacher': '甲', 'name': '離散', 'hours': 3},
    {'teacher': '乙', 'name': '視窗', 'hours': 3},
    {'teacher': '乙', 'name': '科學', 'hours': 3},
    {'teacher': '乙', 'name': '系統', 'hours': 3},
    {'teacher': '乙', 'name': '計概', 'hours': 3},
    {'teacher': '丙', 'name': '軟工', 'hours': 3},
    {'teacher': '丙', 'name': '行動', 'hours': 3},
    {'teacher': '丙', 'name': '網路', 'hours': 3},
    {'teacher': '丁', 'name': '媒體', 'hours': 3},
    {'teacher': '丁', 'name': '工數', 'hours': 3},
    {'teacher': '丁', 'name': '動畫', 'hours': 3},
    {'teacher': '丁', 'name': '電子', 'hours': 4},
    {'teacher': '丁', 'name': '嵌入', 'hours': 3},
    {'teacher': '戊', 'name': '網站', 'hours': 3},
    {'teacher': '戊', 'name': '網頁', 'hours': 3},
    {'teacher': '戊', 'name': '演算', 'hours': 3},
    {'teacher': '戊', 'name': '結構', 'hours': 3},
    {'teacher': '戊', 'name': '智慧', 'hours': 3}
]

teachers = ['甲', '乙', '丙', '丁', '戊']

rooms = ['A', 'B']

slots = [
    'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
    'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
    'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
    'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
    'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
    'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
    'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27',
    'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
    'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47',
    'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]


def assign_schedule():
    schedule = {room: {slot: [] for slot in slots} for room in rooms}

    random_courses = courses.copy()
    random.shuffle(random_courses)

    for course in random_courses:
        if course['hours'] == -1:
            continue
        teacher = course['teacher']
        name = course['name']
        hours = course['hours']

        available_slots = [(room, slot) for room in rooms for slot in slots if len(schedule[room][slot]) < 5]
        random.shuffle(available_slots)

        for _ in range(hours):
            if not available_slots:
                print("課程無法排入時間表")
                return schedule
            room, slot = available_slots.pop()
            schedule[room][slot].append({'teacher': teacher, 'name': name})

    return schedule


def print_schedule(schedule):
    for room in rooms:
        print(f"教室 {room}:")
        for slot in slots:
            print(f"  {slot}:", end="")
            for course in schedule[room][slot]:
                print(f" [{course['teacher']} {course['name']}]", end="")
            print()
        print()


if __name__ == "__main__":
    schedule = assign_schedule()
    print_schedule(schedule)
