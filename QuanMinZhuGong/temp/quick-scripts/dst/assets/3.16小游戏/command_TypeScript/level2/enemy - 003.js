
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level2/enemy - 003.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, 'eacdaUcae9GSpLgd0h4kvHb', 'enemy - 003');
// 3.16小游戏/command_TypeScript/level2/enemy - 003.ts

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
            main_2.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_3.prototype.onCollisionEnter = function (other, self) {
        if (main_2.default.minion_attack == true) {
            main_2.default.main_hp -= 5;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_3.prototype.onCollisionExit = function (other) {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDJcXGVuZW15IC0gMDAzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7QUFDQSxvQkFBb0I7QUFDcEIsd0VBQXdFO0FBQ3hFLG1CQUFtQjtBQUNuQixrRkFBa0Y7QUFDbEYsOEJBQThCO0FBQzlCLGtGQUFrRjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBRTVFLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBQzFDLCtCQUEyQjtBQUszQjtJQUFxQywyQkFBWTtJQUFqRDtRQUFBLHFFQThFQztRQXpFSSxXQUFLLEdBQUcsQ0FBQyxDQUFDOztJQXlFZixDQUFDO0lBeEVHLHdCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFFM0IsQ0FBQztJQUdELHVCQUFLLEdBQUw7UUFDSSxJQUFJLENBQUMsUUFBUSxDQUFDO1lBQ1YsY0FBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUV2QixJQUFHLGNBQU0sQ0FBQyxhQUFhLElBQUksSUFBSSxFQUFFO1lBQzdCLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxDQUFDO1lBQ3BCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDO1lBQzVDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBR0wsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUlsQixjQUFNLENBQUMsYUFBYSxHQUFHLEtBQUssQ0FBQztRQUM3QixJQUFJLGNBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxjQUFNLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQzNEO0lBRUosQ0FBQztJQUdELHdCQUFNLEdBQU4sVUFBUSxFQUFFO1FBRU4sSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDO1FBQzFDLElBQUksSUFBSSxDQUFDLEtBQUssSUFBSSxDQUFDLEVBQUU7WUFDakIsSUFBSSxLQUFLLENBQUMsTUFBTSxJQUFJLEtBQUssRUFBRTtnQkFFdkIsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFHLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pFLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUUsZUFBZSxDQUFDLENBQUE7Z0JBQ3JDLEtBQUssQ0FBQyxjQUFjLENBQUMsR0FBRyxFQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUM3QixJQUFJLENBQUMsS0FBSyxFQUFFLENBQUM7YUFDaEI7U0FDSjtRQUVELElBQUksS0FBSyxDQUFDLE1BQU0sSUFBSSxLQUFLLEVBQUU7WUFDdkIsSUFBSSxjQUFNLENBQUMsYUFBYSxJQUFLLElBQUksRUFBRTtnQkFDL0IsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFJLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFFLENBQUM7YUFHaEY7aUJBQU07Z0JBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFLLElBQUksQ0FBQyxRQUFRLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsRUFBSTtvQkFDdkcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7aUJBQzNFO2FBRUo7U0FDUDtJQUdMLENBQUM7SUE3RWdCLE9BQU87UUFEM0IsT0FBTztPQUNhLE9BQU8sQ0E4RTNCO0lBQUQsY0FBQztDQTlFRCxBQThFQyxDQTlFb0MsRUFBRSxDQUFDLFNBQVMsR0E4RWhEO2tCQTlFb0IsT0FBTyIsImZpbGUiOiIiLCJzb3VyY2VSb290IjoiLyIsInNvdXJjZXNDb250ZW50IjpbIlxyXG4vLyBMZWFybiBUeXBlU2NyaXB0OlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy90eXBlc2NyaXB0Lmh0bWxcclxuLy8gTGVhcm4gQXR0cmlidXRlOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9yZWZlcmVuY2UvYXR0cmlidXRlcy5odG1sXHJcbi8vIExlYXJuIGxpZmUtY3ljbGUgY2FsbGJhY2tzOlxyXG4vLyAgLSBodHRwczovL2RvY3MuY29jb3MuY29tL2NyZWF0b3IvbWFudWFsL2VuL3NjcmlwdGluZy9saWZlLWN5Y2xlLWNhbGxiYWNrcy5odG1sXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuaW1wb3J0IG1haW5fMSBmcm9tIFwiLi9tYWluXCJcclxuXHJcblxyXG5cclxuQGNjY2xhc3NcclxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgZW5lbXlfMyBleHRlbmRzIGNjLkNvbXBvbmVudCB7XHJcbiAgIFxyXG4gICAgLy8gTElGRS1DWUNMRSBDQUxMQkFDS1M6XHJcbiAgICBtaW5pb25feDtcclxuICAgIFxyXG4gICAgIGNvdW50ID0gMDtcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuXHJcbiAgICB9XHJcbiAgICBcclxuXHJcbiAgICBzdGFydCAoKSB7XHJcbiAgICAgICAgdGhpcy5zY2hlZHVsZSgoKSA9PiB7XHJcbiAgICAgICAgICAgIG1haW5fMS5taW5pb25fYXR0YWNrID0gdHJ1ZTtcclxuICAgICAgICB9LDEpO1xyXG5cclxuICAgICAgICB0aGlzLm1pbmlvbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICB9XHJcbiAgICBcclxuICAgIG9uQ29sbGlzaW9uRW50ZXIob3RoZXIsc2VsZil7XHJcbiAgICAgICBcclxuICAgICAgICBpZihtYWluXzEubWluaW9uX2F0dGFjayA9PSB0cnVlKSB7XHJcbiAgICAgICAgICAgIG1haW5fMS5tYWluX2hwIC09IDU7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi01XCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL3h5L21haW5fZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UyLmdldENvbXBvbmVudChjYy5MYWJlbCkuc3RyaW5nID0gXCJcIjtcclxuICAgICAgICB9XHJcbiAgICAgIFxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICBcclxuICAgICAgICBcclxuICAgICAgXHJcbiAgICAgICBtYWluXzEubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgICAgaWYoIG1haW5fMS5tYWluX2hwIDw9IDApIHtcclxuICAgICAgICBvdGhlci5ub2RlLmFjdGl2ZSA9IGZhbHNlO1xyXG4gICAgICAgIGxldCBsb3NlID0gY2MuZmluZChcIkNhbnZhcy9iai9mYWlsXCIpO1xyXG4gICAgICAgIGxvc2UuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZSggbWFpbl8xLm1haW5faHAsIDE5KTtcclxuICAgICAgIH1cclxuICAgICAgIFxyXG4gICAgfVxyXG5cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgIFxyXG4gICAgICAgIGxldCBub2RlMSA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi/lsI/prLwyXCIpO1xyXG4gICAgICAgIGlmICh0aGlzLmNvdW50ID09IDApIHtcclxuICAgICAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24oICB0aGlzLm5vZGUucG9zaXRpb24ueCAsIG5vZGUxLnBvc2l0aW9uLnkpOyAgIFxyXG4gICAgICAgICAgICAgICAgbGV0IG5vZGUyID0gY2MuZmluZCAoXCJDYW52YXMvYmovYi9hXCIpXHJcbiAgICAgICAgICAgICAgICBub2RlMi5zZXRDb250ZW50U2l6ZSg0MDAsMjYpO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5jb3VudCsrOyAgICAgICAgXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKG5vZGUxLmFjdGl2ZSA9PSBmYWxzZSkge1xyXG4gICAgICAgICAgICBpZiAobWFpbl8xLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMubm9kZS5zZXRQb3NpdGlvbiggIHRoaXMubm9kZS5wb3NpdGlvbi54ICAtIDEwMDAqZHQsIHRoaXMubm9kZS5wb3NpdGlvbi55ICk7XHJcbiAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA+PSAgdGhpcy5taW5pb25feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPD0gIHRoaXMubWluaW9uX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcbiJdfQ==