
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level7/right3 - 001.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'f0e01u1aPhDp4Dw+P9f7QGg', 'right3 - 001');
// 3.16小游戏/command_TypeScript/level7/right3 - 001.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___003_1 = require("./global - 003");
var enemy_3 = /** @class */ (function (_super) {
    __extends(enemy_3, _super);
    function enemy_3() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_3.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_3.prototype.start = function () {
        this.schedule(function () {
            global___003_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (global___003_1.default.minion_attack == true) {
            global___003_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
        global___003_1.default.minion_attack = false;
        if (global___003_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___003_1.default.main_hp, 19);
        }
    };
    enemy_3.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼2");
        if (this.count == 0) {
            if (node1.active == false) {
                this.node.setPosition(this.node.position.x, node1.position.y);
                var node2 = cc.find("Canvas/bj/b/a");
                node2.setContentSize(400, 26);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (global___003_1.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_3 = __decorate([
        ccclass
    ], enemy_3);
    return enemy_3;
}(cc.Component));
exports.default = enemy_3;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDdcXHJpZ2h0MyAtIDAwMS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLHNCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsc0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLHNCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUMxQyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBRXZCLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsRUFBRyxLQUFLLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqRSxJQUFJLEtBQUssR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFFLGVBQWUsQ0FBQyxDQUFBO2dCQUNyQyxLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2FBQ2hCO1NBQ0o7UUFFRCxJQUFJLEtBQUssQ0FBQyxNQUFNLElBQUksS0FBSyxFQUFFO1lBQ3ZCLElBQUksc0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO2dCQUMvQixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUksSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUUsQ0FBQzthQUdoRjtpQkFBTTtnQkFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO29CQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztpQkFDM0U7YUFFSjtTQUNQO0lBR0wsQ0FBQztJQTdFZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQThFM0I7SUFBRCxjQUFDO0NBOUVELEFBOEVDLENBOUVvQyxFQUFFLENBQUMsU0FBUyxHQThFaEQ7a0JBOUVvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwM1wiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzMgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgICBjb3VudCA9IDA7XHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcblxyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMuc2NoZWR1bGUoKCkgPT4ge1xyXG4gICAgICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IHRydWU7XHJcbiAgICAgICAgfSwxKTtcclxuXHJcbiAgICAgICAgdGhpcy5taW5pb25feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBnbG9iYWwubWFpbl9ocCAtPSA1O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItNVwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvDJcIik7XHJcbiAgICAgICAgaWYgKHRoaXMuY291bnQgPT0gMCkge1xyXG4gICAgICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICwgbm9kZTEucG9zaXRpb24ueSk7ICAgXHJcbiAgICAgICAgICAgICAgICBsZXQgbm9kZTIgPSBjYy5maW5kIChcIkNhbnZhcy9iai9iL2FcIilcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDQwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLmNvdW50Kys7ICAgICAgICBcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgIGlmIChnbG9iYWwubWluaW9uX2F0dGFjayA9PSAgdHJ1ZSkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgIGlmICghKHRoaXMubm9kZS5wb3NpdGlvbi54ID49ICB0aGlzLm1pbmlvbl94ICsgNTApICYmICh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSAgdGhpcy5taW5pb25feCAtIDUwKSkgICB7XHJcbiAgICAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggKyAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgXHJcbiAgICB9XHJcbn1cclxuIl19