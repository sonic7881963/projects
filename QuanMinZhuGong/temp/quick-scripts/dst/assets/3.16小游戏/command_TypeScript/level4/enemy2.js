
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/enemy2.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'b3216Mf/vtC4YnflgAkn1tW', 'enemy2');
// 3.16小游戏/command_TypeScript/level4/enemy2.ts

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
var global_1 = require("./global");
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
            global_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_2.prototype.onCollisionEnter = function (other, self) {
        if (global_1.default.minion_attack == true) {
            global_1.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_2.prototype.onCollisionExit = function (other) {
        global_1.default.minion_attack = false;
        if (global_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global_1.default.main_hp, 19);
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
            if (global_1.default.minion_attack == true) {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXGVuZW15Mi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQyxtQ0FBNkI7QUFLN0I7SUFBcUMsMkJBQVk7SUFBakQ7UUFBQSxxRUE4RUM7UUF6RUksV0FBSyxHQUFHLENBQUMsQ0FBQzs7SUF5RWYsQ0FBQztJQXhFRyx3QkFBTSxHQUFOO1FBQ0ksSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxDQUFDO1FBQ2hELE9BQU8sQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDO0lBRTNCLENBQUM7SUFHRCx1QkFBSyxHQUFMO1FBQ0ksSUFBSSxDQUFDLFFBQVEsQ0FBQztZQUNWLGdCQUFNLENBQUMsYUFBYSxHQUFHLElBQUksQ0FBQztRQUNoQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUM7UUFFTCxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsa0NBQWdCLEdBQWhCLFVBQWlCLEtBQUssRUFBQyxJQUFJO1FBRXZCLElBQUcsZ0JBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGdCQUFNLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLDZCQUE2QixDQUFDLENBQUM7WUFDcEQsTUFBTSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUM1QyxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUM7WUFDdkQsT0FBTyxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxHQUFHLEVBQUUsQ0FBQztTQUM5QztJQUdMLENBQUM7SUFFRCxpQ0FBZSxHQUFmLFVBQWdCLEtBQUs7UUFJakIsZ0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzlCLElBQUksZ0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxnQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUN6QyxJQUFJLElBQUksQ0FBQyxLQUFLLElBQUksQ0FBQyxFQUFFO1lBQ2pCLElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7Z0JBQ3ZCLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUUsZUFBZSxDQUFDLENBQUE7Z0JBQ3JDLEtBQUssQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO2dCQUNwQixLQUFLLENBQUMsY0FBYyxDQUFDLEdBQUcsRUFBQyxFQUFFLENBQUMsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pFLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQzthQUNoQjtTQUNKO1FBRUQsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtZQUN2QixJQUFJLGdCQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtnQkFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7YUFHaEY7aUJBQU07Z0JBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtvQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzNFO2FBRUo7U0FDUDtJQUdMLENBQUM7SUE3RWdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E4RTNCO0lBQUQsY0FBQztDQTlFRCxBQThFQyxDQTlFb0MsRUFBRSxDQUFDLFNBQVMsR0E4RWhEO2tCQTlFb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IGdsb2JhbCBmcm9tIFwiLi9nbG9iYWxcIlxyXG5cclxuXHJcblxyXG5AY2NjbGFzc1xyXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBlbmVteV8yIGV4dGVuZHMgY2MuQ29tcG9uZW50IHtcclxuICAgXHJcbiAgICAvLyBMSUZFLUNZQ0xFIENBTExCQUNLUzpcclxuICAgIG1pbmlvbl94O1xyXG4gICAgXHJcbiAgICAgY291bnQgPSAwO1xyXG4gICAgb25Mb2FkICgpIHtcclxuICAgICAgICB2YXIgbWFuYWdlciA9IGNjLmRpcmVjdG9yLmdldENvbGxpc2lvbk1hbmFnZXIoKTtcclxuICAgICAgICBtYW5hZ2VyLmVuYWJsZWQgPSB0cnVlO1xyXG5cclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgIFxyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gNTtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi9lbmVteV9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTVcIjtcclxuICAgICAgICAgICAgbGV0IGRhbWFnZTIgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZTIuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIlwiO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG4gICAgXHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgIGdsb2JhbC5taW5pb25fYXR0YWNrID0gZmFsc2U7XHJcbiAgICAgICBpZiggZ2xvYmFsLm1haW5faHAgPD0gMCkge1xyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgbGV0IGxvc2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2ZhaWxcIik7XHJcbiAgICAgICAgbG9zZS5hY3RpdmUgPSB0cnVlO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKCBnbG9iYWwubWFpbl9ocCwgMTkpO1xyXG4gICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG5cclxuICAgIHVwZGF0ZSAoZHQpIHtcclxuICAgICAgXHJcbiAgICAgICAgbGV0IG5vZGUxID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL+Wwj+msvFwiKTtcclxuICAgICAgICBpZiAodGhpcy5jb3VudCA9PSAwKSB7XHJcbiAgICAgICAgICAgIGlmIChub2RlMS5hY3RpdmUgPT0gZmFsc2UpIHtcclxuICAgICAgICAgICAgICAgIGxldCBub2RlMiA9IGNjLmZpbmQgKFwiQ2FudmFzL2JqL2IvYVwiKVxyXG4gICAgICAgICAgICAgICAgbm9kZTIuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIG5vZGUyLnNldENvbnRlbnRTaXplKDIwMCwyNik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAsIG5vZGUxLnBvc2l0aW9uLnkpOyAgIFxyXG4gICAgICAgICAgICAgICAgdGhpcy5jb3VudCsrOyAgICAgICAgXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSAgdGhpcy5taW5pb25feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gIHRoaXMubWluaW9uX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcbiJdfQ==