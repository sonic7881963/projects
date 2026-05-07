
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/enemy - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '8dbe6quYttEIJvOHgQwTjhz', 'enemy - 002');
// 3.16小游戏/command_TypeScript/level2/enemy - 002.ts

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
var main_2 = require("./main");
var enemy_2 = /** @class */ (function (_super) {
    __extends(enemy_2, _super);
    function enemy_2() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.count = 0;
        return _this;
    }
    enemy_2.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_2.prototype.start = function () {
        this.schedule(function () {
            main_2.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_2.prototype.onCollisionEnter = function (other, self) {
        if (main_2.default.minion_attack == true) {
            main_2.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_2.prototype.onCollisionExit = function (other) {
        main_2.default.minion_attack = false;
        if (main_2.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(main_2.default.main_hp, 19);
        }
    };
    enemy_2.prototype.update = function (dt) {
        var node1 = cc.find("Canvas/bj/kuan/小鬼");
        if (this.count == 0) {
            if (node1.active == false) {
                var node2 = cc.find("Canvas/bj/b/a");
                node2.active = true;
                node2.setContentSize(200, 26);
                this.node.setPosition(this.node.position.x, node1.position.y);
                this.count++;
            }
        }
        if (node1.active == false) {
            if (main_2.default.minion_attack == true) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
            else {
                if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                    this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
                }
            }
        }
    };
    enemy_2 = __decorate([
        ccclass
    ], enemy_2);
    return enemy_2;
}(cc.Component));
exports.default = enemy_2;

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGVuZW15IC0gMDAyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtCQUEyQjtBQUszQjtJQUFxQywyQkFBWTtJQUFqRDtRQUFBLHFFQThFQztRQXpFSSxXQUFLLEdBQUcsQ0FBQyxDQUFDOztJQXlFZixDQUFDO0lBeEVHLHdCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFFM0IsQ0FBQztJQUdELHVCQUFLLEdBQUw7UUFDSSxJQUFJLENBQUMsUUFBUSxDQUFDO1lBQ1YsY0FBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUV2QixJQUFHLGNBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxDQUFDO1lBQ3BCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO1lBQzVDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBR0wsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUlsQixjQUFNLENBQUMsYUFBYSxHQUFHLEtBQUssQ0FBQztRQUM3QixJQUFJLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxjQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHdCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDO1FBQ3pDLElBQUksSUFBSSxDQUFDLEtBQUssSUFBSSxDQUFDLEVBQUU7WUFDakIsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtnQkFDdkIsSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBRSxlQUFlLENBQUMsQ0FBQTtnQkFDckMsS0FBSyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7Z0JBQ3BCLEtBQUssQ0FBQyxjQUFjLENBQUMsR0FBRyxFQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUM3QixJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEVBQUcsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDakUsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2FBQ2hCO1NBQ0o7UUFFRCxJQUFJLEtBQUssQ0FBQyxNQUFNLElBQUksS0FBSyxFQUFFO1lBQ3ZCLElBQUksY0FBTSxDQUFDLGFBQWEsSUFBSyxJQUFJLEVBQUU7Z0JBQy9CLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBSSxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBRSxDQUFDO2FBR2hGO2lCQUFNO2dCQUNILElBQUksQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUk7b0JBQ3ZHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2lCQUMzRTthQUVKO1NBQ1A7SUFHTCxDQUFDO0lBN0VnQixPQUFPO1FBRDNCLE9BQU87T0FDYSxPQUFPLENBOEUzQjtJQUFELGNBQUM7Q0E5RUQsQUE4RUMsQ0E5RW9DLEVBQUUsQ0FBQyxTQUFTLEdBOEVoRDtrQkE5RW9CLE9BQU8iLCJmaWxlIjoiIiwic291cmNlUm9vdCI6Ii8iLCJzb3VyY2VzQ29udGVudCI6WyJcclxuLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5cclxuY29uc3Qge2NjY2xhc3MsIHByb3BlcnR5fSA9IGNjLl9kZWNvcmF0b3I7XHJcbmltcG9ydCBtYWluXzEgZnJvbSBcIi4vbWFpblwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzIgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgICBjb3VudCA9IDA7XHJcbiAgICBvbkxvYWQgKCkge1xyXG4gICAgICAgIHZhciBtYW5hZ2VyID0gY2MuZGlyZWN0b3IuZ2V0Q29sbGlzaW9uTWFuYWdlcigpO1xyXG4gICAgICAgIG1hbmFnZXIuZW5hYmxlZCA9IHRydWU7XHJcblxyXG4gICAgfVxyXG4gICAgXHJcblxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICAgIHRoaXMuc2NoZWR1bGUoKCkgPT4ge1xyXG4gICAgICAgICAgICBtYWluXzEubWluaW9uX2F0dGFjayA9IHRydWU7XHJcbiAgICAgICAgfSwxKTtcclxuXHJcbiAgICAgICAgdGhpcy5taW5pb25feCA9IHRoaXMubm9kZS5wb3NpdGlvbi54O1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICAgXHJcbiAgICAgICAgaWYobWFpbl8xLm1pbmlvbl9hdHRhY2sgPT0gdHJ1ZSkge1xyXG4gICAgICAgICAgICBtYWluXzEubWFpbl9ocCAtPSA1O1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCItNVwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICBcclxuICAgICAgICBcclxuICAgICAgXHJcbiAgICAgICBtYWluXzEubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgaWYoIG1haW5fMS5tYWluX2hwIDw9IDApIHtcclxuICAgICAgICBvdGhlci5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgIGxldCBsb3NlID0gY2MuZmluZChcIkNhbnZhcy9iai9mYWlsXCIpO1xyXG4gICAgICAgIGxvc2UuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZSggbWFpbl8xLm1haW5faHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgIFxyXG4gICAgICAgIGxldCBub2RlMSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi/lsI/prLxcIik7XHJcbiAgICAgICAgaWYgKHRoaXMuY291bnQgPT0gMCkge1xyXG4gICAgICAgICAgICBpZiAobm9kZTEuYWN0aXZlID09IGZhbHNlKSB7XHJcbiAgICAgICAgICAgICAgICBsZXQgbm9kZTIgPSBjYy5maW5kIChcIkNhbnZhcy9iai9iL2FcIilcclxuICAgICAgICAgICAgICAgIG5vZGUyLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICBub2RlMi5zZXRDb250ZW50U2l6ZSgyMDAsMjYpO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggLCBub2RlMS5wb3NpdGlvbi55KTsgICBcclxuICAgICAgICAgICAgICAgIHRoaXMuY291bnQrKzsgICAgICAgIFxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIGlmIChub2RlMS5hY3RpdmUgPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgaWYgKG1haW5fMS5taW5pb25fYXR0YWNrID09ICB0cnVlKSB7XHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAgLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSApO1xyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbih0aGlzLm5vZGUucG9zaXRpb24ueCArIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55KTtcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBcclxuICAgIH1cclxufVxyXG4iXX0=